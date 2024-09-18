import os
import time
from dataclasses import dataclass

import napari
import numpy as np
import pyclesperanto as cle
import skimage.io
import skimage.io
from magicgui import magicgui
from napari.types import ImageData, LabelsData
from skimage.filters import threshold_multiotsu
from tqdm import tqdm


# Constants


## Classes
@dataclass(kw_only=True)
class SegmentationSettings:
    otsu_sigma: float = 0.6
    particle_mask_sigma: float = 0.1
    particle_n_erosions: int = 2
    particle_enlarge_radius: int = 1
    smooth_labels_radius: int = 2

    def __str__(self):
        return (
                "Segmentation settings:" +
                "\n Otsu thresholding: Gaussian smoothing sigma: {:.2f}".format(self.otsu_sigma) +
                "\n Particle mask: Gaussian smoothing sigma: {:.2f}".format(self.particle_mask_sigma) +
                "\n Particle mask: Erosion iterations: {}".format(self.particle_n_erosions) +
                "\n Particle mask: Dilation radius: {}".format(self.particle_enlarge_radius) +
                "\n Mask postprocessing: Smoothing radius: {}".format(self.smooth_labels_radius)
        )


## Scan io
def load_stack(path: str,
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

    path = path + "Slices/"
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
        tuple: Tuple containing the voxel size and the scan dimensions in mm."""
    with open(path + "unireconstruction.xml", "r") as f:
        text = f.read()
    voxel_size = float(text.split("<voxelSize X=\"")[1].split("\"")[0])
    return voxel_size


def save_scan(scan, path, to_png=False):
    print("Saving image to: ", path)
    if not os.path.isdir(path):
        os.mkdir(path)
    for i, img in tqdm(enumerate(scan)):
        filename = path + "slice{:04d}.{:s}".format(i, 'png' if to_png else 'tif')
        skimage.io.imsave(filename, img, check_contrast=False)


def save_mask(scan, path):
    print("Saving image to: ", path)
    if not os.path.isdir(path):
        os.mkdir(path)
    for i, img in tqdm(enumerate(scan)):
        filename = path + "slice{:04d}.pdf".format(i)
        skimage.io.imsave(filename, img, check_contrast=False)


def reslice(scan: np.ndarray,
            axis: tuple[int, int, int] = (1, 0, 2)) -> np.ndarray:
    return np.transpose(scan, axis)


def show_in_napari(img, *labels):
    viewer = napari.Viewer()
    viewer.add_image(img)
    for label in labels:
        viewer.add_labels(label)


def divide_scan(scan, size_gb: float = 1):
    """Returns the indexes that split an array into parts of roughly equal size less than size_gb GB."""
    n_parts = int(np.ceil(scan.nbytes / size_gb / 1e9))
    n_slices, _, _ = scan.shape
    part_size = n_slices // n_parts
    rest = n_slices % n_parts
    part_sizes = [0] + [part_size] * n_parts
    part_sizes[-1] += rest
    part_sizes = np.cumsum(part_sizes)
    return part_sizes


def particle_segmentation(im: np.ndarray[np.uint16],
                          settings: SegmentationSettings = SegmentationSettings,
                          low_memory_mode: int = True) -> LabelsData | np.ndarray[np.int32]:
    """Segment particles in a 3D image using a combination of gaussian blur, otsu thresholding, and morphological operations.

    If low_memory_mode==True, the function will not apply erosion and subsequent masked voronoi labeling to the mask.
    The result is a binary mask with the particles labeled as 1 and the background as 0. This does not matter for general analysis
    of the particle neighborhood, but prevents statistics on the particles themselves. If more memory is available
    (10x the size of the image as returned by the np.nbytes command), set low_memory_mode==False to improve segmentation results.

    Args:
        im (ImageData): 3D image data as a np.ndarray.
        sigma (float): Sigma for the gaussian blur before thresholding.
        n_erosions (int): Number of erosions to apply to the mask.
        dilation_radius (int): Radius for the dilation of the mask.
        low_memory_mode (bool): If True, the function will not apply erosion and subsequent masked voronoi labeling to the mask.

    Returns:
        LabelsData: Binary mask with the particles labeled as 1 and the background as 0 if low_memory_mode==True.
                    Otherwise, a connected component labeled mask.
    """
    sigma = settings.particle_mask_sigma
    n_erosions = settings.particle_n_erosions
    dilation_radius = settings.particle_enlarge_radius

    if low_memory_mode:
        smoothed_gpu = cle.create_like(im, dtype=np.float32)
        mask_gpu = cle.create_like(im, dtype=np.uint32)

        # apply gaussian blur and otsu threshold
        cle.gaussian_blur(im, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
        cle.threshold_otsu(smoothed_gpu, output_image=mask_gpu)
        del smoothed_gpu

        cle.dilate_labels(mask_gpu, output_image=mask_gpu, radius=dilation_radius)
        cle.erode_connected_labels(mask_gpu, output_image=mask_gpu, radius=1)

        mask = cle.pull(mask_gpu)
        del mask_gpu

        return mask

    else:
        # allocate memory on the gpu
        smoothed_gpu = cle.create_like(im, dtype=np.float32)
        mask_gpu = cle.create_like(im, dtype=np.uint32)

        # apply gaussian blur and otsu threshold
        cle.gaussian_blur(im, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
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


def otsu_mask(scan: np.ndarray[np.uint32],
              settings: SegmentationSettings = SegmentationSettings()) -> np.ndarray[np.uint8]:
    """Segment a 3D image using a combination of gaussian blur and otsu thresholding into three classes.

    Args:
        scan (ImageData): 3D image data as a np.ndarray.
        sigma (float): Sigma for the gaussian blur before thresholding.

    Returns:
        np.ndarray: Mask with three classes: 1 for air, 2 for polymer, and 3 for particles."""

    # smoothing on GPU
    sigma = settings.otsu_sigma

    scan_gpu = cle.push(scan)
    smoothed_gpu = cle.create_like(scan_gpu, dtype=np.float32)
    cle.gaussian_blur(scan_gpu, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
    im = cle.pull(smoothed_gpu)
    del smoothed_gpu

    # multi-otsu thresholding
    th1, th2 = threshold_multiotsu(im, classes=3)

    # create mask
    mask: np.ndarray[np.uint8] = np.zeros_like(im, dtype=np.uint8)  # type:ignore
    mask[im <= th1] = 1
    mask[(im > th1) & (im < th2)] = 2
    mask[im >= th2] = 3

    return mask


def segment_scan(scan: np.ndarray[np.uint16],
                 settings: SegmentationSettings = SegmentationSettings(),
                 low_memory_mode: bool = True) -> np.ndarray[np.uint8]:
    """Segment a 3D image into air, polymer, and particles using a combined approach of smoothing and otsu thresholding.

    Args:
        scan (ImageData): 3D image data as a np.ndarray.
        settings (SegmentationSettings): Segmentation settings.
        low_memory_mode (bool): If True, the function will not apply erosion and subsequent masked voronoi labeling to the mask.

    Returns:
        np.ndarray: Mask with three classes: 1 for air, 2 for polymer, and 3 for particles.
    """
    # More precise particle segmentation (with less smoothing)

    if low_memory_mode:
        fine_particle_mask = particle_segmentation(im=scan,
                                                   settings=settings,
                                                   low_memory_mode=True)
    else:
        fine_particle_mask = particle_segmentation(im=scan,
                                                   settings=settings,
                                                   low_memory_mode=False)

    # Otsu thresholding
    mask: np.ndarray[np.uint8] = otsu_mask(scan,
                     settings=settings)

    # overwrite particles (and adjacent regions) with better particle mask
    mask[fine_particle_mask > 0] = 3

    if settings.smooth_labels_radius > 0:
        mask_gpu = cle.push(mask)
        cle.smooth_labels(mask_gpu, output_image=mask_gpu, radius=settings.smooth_labels_radius)
        mask = cle.pull(mask_gpu).astype(np.uint8)
        del mask_gpu

    return mask


def adjust_segmentation_parameters_on_subset(scan: ImageData,
                                             subset_size: int = 30,
                                             low_memory_mode: float = 0.5) -> None:
    """Opens a napari viewer with a subset of the scan and a GUI to adjust the segmentation parameters.

    The parameters that can be adjusted are:
    - otsu_sigma: Sigma for the gaussian blur before otsu thresholding.
    - particle_enlarge_radius: Radius for the dilation of the particle mask.
    - particle_mask_sigma: Sigma for the gaussian blur before the particle mask is created.
    - smooth_labels_radius: Radius for the smoothing of the labels.

    """

    @magicgui(auto_call=True)
    def interactive_segmentation(scan: ImageData,
                                 otsu_sigma: float = 0.6,
                                 particle_mask_sigma: float = 0.1,
                                 particle_n_erosions: int = 2,
                                 particle_enlarge_radius: int = 1,
                                 smooth_labels_radius: int = 2) -> LabelsData:
        """Segment the scan using the given parameters. If low_memory_mode==True, n_erosions is set to 0.

        Args:
            scan (ImageData): 3D image data as a np.ndarray.
            otsu_sigma (float): Sigma for the gaussian blur before thresholding.
            particle_mask_sigma (float): Sigma for the gaussian blur before the particle mask is created.
            n_erosions (int): Number of erosions to apply to the mask.
            particle_enlarge_radius (int): Radius for the dilation of the particle mask.
            smooth_labels_radius (int): Radius for the smoothing of the labels.

        Returns:
            LabelsData: Mask where air is labeled as 1, polymer as 2, and particles as 3."""
        settings = SegmentationSettings(otsu_sigma=otsu_sigma,
                                        particle_mask_sigma=particle_mask_sigma,
                                        particle_n_erosions=particle_n_erosions,
                                        particle_enlarge_radius=particle_enlarge_radius,
                                        smooth_labels_radius=smooth_labels_radius)

        return segment_scan(scan, settings=settings, low_memory_mode=low_memory_mode) # type:ignore

    n_slices, _, _ = scan.shape
    subscan = scan[n_slices // 2 - subset_size // 2:n_slices // 2 + subset_size // 2, :, :]

    viewer = napari.Viewer()
    viewer.add_image(subscan)  # type:ignore
    viewer.window.add_dock_widget(interactive_segmentation, area='right')
