import napari
import numpy as np
import os
import pyclesperanto as cle
import skimage.io
import skimage.io
import time
from dataclasses import dataclass
from magicgui import magicgui
from napari.types import ImageData, LabelsData
from tqdm import tqdm

from mt.utils import get_transpose_order


## Data class for the segmentation settings
@dataclass(kw_only=True)
class SegmentationSettings:
    """Data class for the segmentation settings.

    Attributes:
        air_mask_sigma (float): Sigma for the Gaussian blur before air-polymer thresholding.
        air_thresh (tuple): Threshold for the air-polymer mask.
        particle_mask_sigma (float): Sigma for the Gaussian blur before the particle thresholding.
        particle_n_erosions (int): Number of erosions to apply to the mask.
        particle_enlarge_radius (int): Radius for the dilation of the particle mask.
        smooth_labels_radius (int): Radius for the smoothing of the labels."""

    air_mask_sigma: float = 1.6
    air_thresh: tuple[int, int] = (0, 0)
    particle_mask_sigma: float = 0.1
    particle_n_erosions: int = 2
    particle_enlarge_radius: int = 1
    smooth_labels_radius: int = 2

    def __str__(self):
        return (
                "air_mask_sigma = {}".format(self.air_mask_sigma) +
                "\nair_threshold = {}".format(self.air_thresh) +
                "\nparticle_mask_sigma = {}".format(self.particle_mask_sigma) +
                "\nparticle_n_erosions = {}".format(self.particle_n_erosions) +
                "\nparticle_enlarge_radius = {}".format(self.particle_enlarge_radius) +
                "\nsmooth_labels_radius = {}".format(self.smooth_labels_radius)
        )


## Scan io
def load_stack(path: str,
               folder: str = "Slices",
               file_extension: str = "tif",
               image_range: tuple | None = None,
               logging: bool = False) -> np.ndarray:
    """Loads a stack of images from a folder and returns them as a numpy array.

    It is possible to only load a subset of the images by specifying the image_range.
    The logging parameter can be used to print information about the loading process, like time and memory usage.


    Args:
        path (str): Path to the folder containing the images.
        file_extension (str): File extension of the images.
        image_range (tuple): Range of images to load.
        logging (bool): If True, prints information about the loading process.

    Returns:
        np.ndarray: Stack of images.

    Raises:
        ValueError: If no images are found in the folder."""

    path = path + folder + "/"
    start = time.time()
    logging and print("Loading images from: ", path)
    files = [path + file for file in os.listdir(path) if file.endswith(file_extension)]
    if len(files) == 0:
        raise ValueError("No images found in path matching the given extension.")
    if image_range is None:
        image_range = (0, len(files))
    imgs = skimage.io.imread_collection(files[image_range[0]:image_range[1]])
    scan = np.array(imgs)
    logging and print(
            "Loaded stack with shape {} and a size of {:.2f} GB in {:.2f} s.".format(scan.shape,
                                                                                     scan.nbytes / 1e9,
                                                                                     time.time() - start))
    return scan


def read_scan_properties(path: str) -> float:
    """Reads scan parameters from the metadata files in the path folder.

    Currently, only the voxel size is read from the unireconstruction.xml file.

    Args:
        path (str): Path to the folder containing the metadata files.

    Returns:
        float: Voxel size in um."""
    with open(path + "unireconstruction.xml", "r") as f:
        text = f.read()
    voxel_size = float(text.split("<voxelSize X=\"")[1].split("\"")[0])
    return voxel_size


def save_scan(scan: np.ndarray,
              path: str,
              to_png=False):
    """Saves a stack of images to a folder.

    Args:
        scan (np.ndarray): Stack of images.
        path (str): Path to the folder where the images will be saved.
        to_png (bool): If True, the images will be saved as png files. Otherwise, they will be saved as tif files."""
    print("Saving image to: ", path)
    if not os.path.isdir(path):
        os.mkdir(path)
    for i, img in tqdm(enumerate(scan)):
        filename = path + "slice{:04d}.{:s}".format(i, 'png' if to_png else 'tif')
        skimage.io.imsave(filename, img, check_contrast=False)


def reslice(scan: np.ndarray,
            axis: str = "z") -> np.ndarray:
    """Reslice a 3D image along the given axis.

    Args:
        scan (np.ndarray): 3D image data.
        axis (str): Axis along which to reslice the image.

    Returns:
        np.ndarray: Resliced image."""
    order = get_transpose_order(scan, axis)
    return np.transpose(scan, order)


def particle_segmentation(im: np.ndarray[np.uint16],
                          settings: SegmentationSettings = SegmentationSettings) -> LabelsData | np.ndarray[np.int32]:
    """Segment particles in a 3D image using a combination of gaussian blur, otsu thresholding, and morphological operations.

    The segmentation takes place on the GPU using the pyclesperanto library. All images are allocated on the GPU and
    only the final mask is transferred back to the CPU.

    Args:
        im (ImageData): 3D image data as a np.ndarray.
        settings (SegmentationSettings): Segmentation settings.

    Returns:
        LabelsData: Binary mask with the particles labeled as 1 and the background as 0 if low_memory_mode==True.
                    Otherwise, a connected component labeled mask.
    """

    # Set settings based on the SegmentationSettings object
    sigma = settings.particle_mask_sigma
    n_erosions = settings.particle_n_erosions
    dilation_radius = settings.particle_enlarge_radius
    particle_thresh = 0
    # If a particle threshold is set, use it
    if settings.air_thresh[1] != 0:
        particle_thresh = settings.air_thresh[1]

    # allocate memory on the gpu
    smoothed_gpu = cle.create_like(im, dtype=np.float32)
    mask_gpu = cle.create_like(im, dtype=np.uint16)

    # apply Gaussian blur and do Otsu or manual thresholding
    cle.gaussian_blur(im, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
    if particle_thresh != 0:
        cle.greater_constant(smoothed_gpu, output_image=mask_gpu, scalar=particle_thresh)
    else:
        cle.threshold_otsu(smoothed_gpu, output_image=mask_gpu)
    del smoothed_gpu

    # apply morphological operations
    original_mask = cle.copy(mask_gpu)
    for _ in range(n_erosions):
        cle.erode_labels(mask_gpu, output_image=mask_gpu, radius=1)
    cle.masked_voronoi_labeling(mask_gpu, output_image=mask_gpu, mask=original_mask)
    del original_mask
    cle.dilate_labels(mask_gpu, output_image=mask_gpu, radius=dilation_radius)
    cle.erode_connected_labels(mask_gpu, output_image=mask_gpu, radius=1)

    mask = cle.pull(mask_gpu)
    del mask_gpu

    return mask


def voronoi_tesselation(im: np.ndarray[np.uint16],
                        settings: SegmentationSettings = SegmentationSettings()) -> np.ndarray[np.uint16]:
    """Create a voronoi tesselation of the particles in a 3D image.

    The image is blurred with a Gaussian filter, eroded to create seeds, and then the
    voronoi tesselation is computed using the pyclesperanto library.

    Args:
        im (ImageData): 3D image data as a np.ndarray.
        settings (SegmentationSettings): Segmentation settings.

    Returns:
        np.ndarray: Voronoi tesselation of the particles."""

    # Set settings based on the SegmentationSettings object
    sigma = settings.particle_mask_sigma
    n_erosions = settings.particle_n_erosions
    particle_thresh = 0
    if settings.air_thresh[1] != 0:
        particle_thresh = settings.air_thresh[1]

    # allocate memory on the gpu
    smoothed_gpu = cle.create_like(im, dtype=np.float32)
    mask_gpu = cle.create_like(im, dtype=np.uint16)
    v_tess_gpu = cle.create_like(im, dtype=np.uint16)

    # apply gaussian blur and otsu threshold
    cle.gaussian_blur(im, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
    if particle_thresh != 0:
        cle.greater_constant(smoothed_gpu, output_image=mask_gpu, scalar=particle_thresh)
    else:
        cle.threshold_otsu(smoothed_gpu, output_image=mask_gpu)
    del smoothed_gpu

    # apply morphological operations
    for _ in range(n_erosions):
        cle.erode_labels(mask_gpu, output_image=mask_gpu, radius=1)

    # compute voronoi tesselation
    cle.voronoi_labeling(mask_gpu, v_tess_gpu)
    v_tess = cle.pull(v_tess_gpu)
    del mask_gpu, v_tess_gpu

    return v_tess


def air_segmentation(scan: np.ndarray[np.uint16],
                     settings: SegmentationSettings = SegmentationSettings()) -> np.ndarray[np.uint8]:
    """Segment the air in a 3D image using a combination of gaussian blur and otsu thresholding.

    Args:
        scan (ImageData): 3D image data as a np.ndarray.
        settings (SegmentationSettings): Segmentation settings.

    Returns:
        np.ndarray: Binary mask with the air labeled as 1 and the rest as 0."""

    # Set settings based on the SegmentationSettings object
    sigma = settings.air_mask_sigma
    air_thresh = settings.air_thresh[0]

    # allocate memory on the gpu
    smoothed_gpu = cle.create_like(scan, dtype=np.float32)
    mask_gpu = cle.create_like(scan, dtype=np.uint16)

    # apply gaussian blur and threshold
    cle.gaussian_blur(scan, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
    cle.smaller_constant(smoothed_gpu, output_image=mask_gpu, scalar=air_thresh)
    mask = cle.pull(mask_gpu).astype(np.uint8)
    del smoothed_gpu, mask_gpu

    return mask


def segment_scan(scan: np.ndarray[np.uint16],
                 settings: SegmentationSettings = SegmentationSettings()) -> np.ndarray[np.uint8]:
    """Segment a 3D image into air, polymer, and particles using a combined approach of smoothing and otsu thresholding.

    First, the manually thesholded air-polymer mask is applied. This mask is overwritten with the particle mask, which
    replaces a part of the polymer segmentation with particles. Finally, the mask is smoothed using a connected component
    labeling and a smoothing operation.

    Args:
        scan (ImageData): 3D image data as a np.ndarray.
        settings (SegmentationSettings): Segmentation settings.
        low_memory_mode (bool): If True, the function will not apply erosion and subsequent masked voronoi labeling to the mask.

    Returns:
        np.ndarray: Mask with three classes: 1 for air, 2 for polymer, and 3 for particles.
    """

    # More precise particle segmentation (with less smoothing)
    particle_mask = particle_segmentation(im=scan, settings=settings)

    # Manual air thresholding
    air_mask = air_segmentation(scan=scan, settings=settings)

    # overwrite particles (and adjacent regions) with better particle mask
    mask = np.zeros_like(air_mask)
    mask[air_mask == 0] = 2  # Polymer
    mask[air_mask == 1] = 1  # Air
    mask[particle_mask > 0] = 3  # Particles

    if settings.smooth_labels_radius > 0:
        mask_gpu = cle.push(mask)
        cle.smooth_labels(mask_gpu, output_image=mask_gpu, radius=settings.smooth_labels_radius)
        mask = cle.pull(mask_gpu).astype(np.uint8)
        del mask_gpu

    return mask


def adjust_segmentation_parameters_on_subset(scan: ImageData,
                                             subset_size: int = 30,
                                             autorun: bool = False,
                                             segment_particles_only: bool = False) -> None:
    """Opens a napari viewer with a subset of the scan and a GUI to adjust the segmentation parameters.

    Args:
        scan (ImageData): 3D image data as a np.ndarray.
        subset_size (int): Size of the subset to display in the viewer. If -1, the whole scan is displayed.
        autorun (bool): If True, the segmentation will be run automatically when the parameters are changed.
        segment_particles_only (bool): If True, the particle segmentation will be displayed in the viewer.
        """

    @magicgui(auto_call=autorun)
    def interactive_segmentation(scan: ImageData,
                                 air_mask_sigma: float = 0.6,
                                 air_threshhold: int = 70,
                                 particle_threshold: int = 150,
                                 particle_mask_sigma: float = 0.1,
                                 particle_n_erosions: int = 2,
                                 particle_enlarge_radius: int = 1,
                                 smooth_labels_radius: int = 2) -> LabelsData:
        settings = SegmentationSettings(air_mask_sigma=air_mask_sigma,
                                        air_thresh=(air_threshhold * 255, particle_threshold * 255),
                                        particle_mask_sigma=particle_mask_sigma,
                                        particle_n_erosions=particle_n_erosions,
                                        particle_enlarge_radius=particle_enlarge_radius,
                                        smooth_labels_radius=smooth_labels_radius)
        return segment_scan(scan, settings=settings)  # type:ignore

    @magicgui(auto_call=autorun)
    def interactive_particle_segmentation(scan: ImageData,
                                          particle_mask_sigma: float = 0.8,
                                          particle_thresh: int = 150,
                                          particle_n_erosions: int = 4,
                                          particle_enlarge_radius: int = 1) -> LabelsData:
        settings = SegmentationSettings(particle_mask_sigma=particle_mask_sigma,
                                        air_thresh=(1 * 255, particle_thresh * 255),
                                        particle_n_erosions=particle_n_erosions,
                                        particle_enlarge_radius=particle_enlarge_radius)
        return particle_segmentation(scan, settings=settings)  # type:ignore

    if subset_size == -1:
        subscan = scan
    else:
        n_slices, _, _ = scan.shape
        subscan = scan[n_slices // 2 - subset_size // 2:n_slices // 2 + subset_size // 2, :, :]

    viewer = napari.Viewer()
    # type:ignore
    if segment_particles_only:
        subscan = np.transpose(subscan, get_transpose_order(subscan, "z"))
        viewer.add_image(subscan)
        viewer.window.add_dock_widget(interactive_particle_segmentation, area='right')
    else:
        viewer.add_image(subscan)
        viewer.window.add_dock_widget(interactive_segmentation, area='right')


def contact_area(image: np.ndarray,
                 label1: int,
                 label2: int) -> int:
    """Counts the number of surfaces of label1 in contact with label2.

    Args:
        image(numpy.ndarray): Array representing image (or image stack).
        label1(int): Value of label 1
        label1(int): Value of label 2

    Returns:
        int: Number of voxels of label1 in contact with label2
    """

    _, bins = skimage.exposure.histogram(image)

    if label1 not in bins or label2 not in bins:
        return 1  # minimal contact area, prevents division by zero

    x_contact_1 = np.logical_and(image[:, :, :-1] == label1, image[:, :, 1:] == label2)
    x_contact_2 = np.logical_and(image[:, :, :-1] == label2, image[:, :, 1:] == label1)
    y_contact_1 = np.logical_and(image[:, :-1, :] == label1, image[:, 1:, :] == label2)
    y_contact_2 = np.logical_and(image[:, :-1, :] == label2, image[:, 1:, :] == label1)
    z_contact_1 = np.logical_and(image[:-1, :, :] == label1, image[1:, :, :] == label2)
    z_contact_2 = np.logical_and(image[:-1, :, :] == label2, image[1:, :, :] == label1)

    contact_voxels = (np.count_nonzero(x_contact_1) +
                      np.count_nonzero(x_contact_2) +
                      np.count_nonzero(y_contact_1) +
                      np.count_nonzero(y_contact_2) +
                      np.count_nonzero(z_contact_1) +
                      np.count_nonzero(z_contact_2))

    return contact_voxels


def get_areas_and_contact(tesselation_2d: np.ndarray,
                          mask_2d: np.ndarray,
                          voxel_area: float,
                          averaging_method: str = "mean",
                          grid: tuple[int, int] = (5, 5)):
    """Compute the mean or median of the voronoi cell areas and the contact percentage of Al-PMMA interface in each region.

    Args:
        tesselation_2d (np.ndarray): 2D voronoi tesselation image.
        mask_2d (np.ndarray): 2D mask image.
        voxel_area (float): Area of a voxel in um^2.
        averaging_method (str): Method to compute the mean or median of the voronoi cell areas.
        grid (tuple): Grid dimensions to divide the image into regions.

    Returns:
        tuple: Tuple containing the mean or median of the voronoi cell areas and the contact percentage of Al-PMMA interface in each region.
    """

    def divide_into_grid(image) -> list[np.ndarray]:
        """Convert a 2D image into a grid of regions."""
        h, w = image.shape
        cell_h = h // grid[0]
        cell_w = w // grid[1]
        regions = []
        for i in range(grid[0]):
            for j in range(grid[1]):
                regions.append(image[i * cell_h:(i + 1) * cell_h, j * cell_w:(j + 1) * cell_w])
        return regions

    def region_cell_areas(regions) -> np.ndarray:
        """For each image region compute the median or mean of the voronoi cell size."""
        areas = []
        for region in regions:
            areas_px, _ = skimage.exposure.histogram(region)
            areas_px = areas_px[areas_px.astype(bool)]  # remove zeros
            areas.append(np.mean(areas_px) if averaging_method == "mean" else np.median(areas_px))
        areas_px = np.array(areas).reshape(*grid)
        areas_um2 = areas_px * voxel_area
        return areas_um2

    def polymer_contact(regions):
        """Returns the contact percentage of Al-PMMA interface in each region."""
        contact_pct = []
        for region in regions:
            region = region[..., np.newaxis]
            air_Al = contact_area(region, 1, 3)
            poly_Al = contact_area(region, 2, 3)
            contact_pct.append(poly_Al / (air_Al + poly_Al))
        contact_pct = np.array(contact_pct).reshape(*grid)
        return contact_pct

    # get means of voronoi cell areas for regions
    regions = divide_into_grid(tesselation_2d)
    areas_um2 = region_cell_areas(regions)

    # get polymer contact percentage for regions
    regions = divide_into_grid(mask_2d)
    contact_percent = polymer_contact(regions)

    return areas_um2, contact_percent
