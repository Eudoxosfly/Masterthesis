import os
import time
from dataclasses import dataclass

import napari
import numpy as np
import pyclesperanto as cle
import skimage.io
from magicgui import magicgui
from napari.types import ImageData, LabelsData
from skimage.filters import threshold_multiotsu


class Scan:
    def __init__(self, path: str):
        self.path: str = path
        self.scan: np.ndarray[np.uint16] | None = None
        self.voxel_size_mm: float | None = None
        self.scan_dimensions_mm: tuple[float, float, float] | None = None
        self.V: float | None = None
        self.slice_orientation: list = ["y", "z", "x"]

    def __getitem__(self, item):
        return self.scan[item]

    def __str__(self):
        return (
                "Scan with shape {} and voxel size {:.2f}."
                .format(self.scan.shape, self.voxel_size_um) +
                "\n Scanned volume: ({:.1f}x{:.1f}x{:.1f}) mm".format(*self.scan_dimensions_mm) +
                "\n Volume: {:.2f} mm^3".format(self.V)

        )

    def calculate_properties(self):
        self.scan_dimensions_mm = self.calc_dimensions()
        self.V = self.calc_volume()

    def calc_dimensions(self):
        h, w, d = self.scan.shape
        return (h * self.voxel_size_mm,
                w * self.voxel_size_mm,
                d * self.voxel_size_mm)

    def calc_volume(self) -> float:
        return float(np.prod(self.scan_dimensions_mm))

    def reslice(self,
                axis: tuple[int, int, int] | str = (1, 0, 2)):
        if isinstance(axis, str):
            orientations: dict[str, int] = {"x": 0, "y": 1, "z": 2}
            np.where(self.slice_orientation == axis)
        self.scan = np.transpose(self.scan, axis)

    def load(self,
             image_range: str | None = None,
             logging: bool = False):
        self.scan = load_stack(path=self.path,
                               image_range=image_range,
                               logging=logging)
        self.voxel_size_mm = read_scan_properties(self.path)

    def try_segmentation_settings(self):
        adjust_segmentation_parameters_on_subset(scan=self.scan,
                                                    subset_size=30,
                                                    low_memory_mode=True)

    def set_segmentation_settings(self):
        self.settings


@dataclass
class SegmentationSettings:
    otsu_sigma: float = 0.6
    particle_mask_sigma: float = 0.1
    n_erosions: int = 2
    particle_enlarge_radius: int = 1
    smooth_labels_radius: int = 2

    def __str__(self):
        return (
            "Segmentation settings:\n" +
            f"Otsu thresholding: Gauss smoothing sigma : {self.otsu_sigma}\n" +
            f"particle segmentation: Gauss smoothing sigma: {self.particle_mask_sigma}\n" +
            f"Particle segmentation: Number of erosions: {self.n_erosions}\n" +
            f"Particle segmentation: Particle dilation radius: {self.particle_enlarge_radius}\n" +
            f"Mask postprocessing: Smoothing radius: {self.smooth_labels_radius}"
        )
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


def particle_segmentation(im: np.ndarray[np.uint16],
                          sigma: float = 0.1,
                          n_erosions: int = 2,
                          dilation_radius: int = 1,
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
              sigma: float = 1) -> np.ndarray[np.uint8]:
    """Segment a 3D image using a combination of gaussian blur and otsu thresholding into three classes.

    Args:
        scan (ImageData): 3D image data as a np.ndarray.
        sigma (float): Sigma for the gaussian blur before thresholding.

    Returns:
        np.ndarray: Mask with three classes: 1 for air, 2 for polymer, and 3 for particles."""

    # smoothing on GPU
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
                 otsu_sigma: float = 1,
                 particle_mask_sigma: float = 0.1,
                 particle_erosions: int = 2,
                 particle_enlarge_radius: int = 1,
                 smooth_labels_radius: int = 0,
                 low_memory_mode: bool = True) -> np.ndarray[np.uint8]:
    # More precise particle segmentation (with less smoothing)
    if low_memory_mode:
        fine_particle_mask = particle_segmentation(im=scan,
                                                   sigma=particle_mask_sigma,
                                                   dilation_radius=particle_enlarge_radius,
                                                   low_memory_mode=True)
    else:
        fine_particle_mask = particle_segmentation(im=scan,
                                                   sigma=particle_mask_sigma,
                                                   n_erosions=particle_erosions,
                                                   dilation_radius=particle_enlarge_radius,
                                                   low_memory_mode=False)

    # Otsu thresholding
    mask = otsu_mask(scan, sigma=otsu_sigma)

    # overwrite particles (and adjacent regions) with better particle mask
    mask[fine_particle_mask > 0] = 3

    if smooth_labels_radius > 0:
        mask_gpu = cle.push(mask)
        cle.smooth_labels(mask_gpu, output_image=mask_gpu, radius=smooth_labels_radius)
        mask = cle.pull(mask_gpu)
        del mask_gpu

    return mask


def adjust_segmentation_parameters_on_subset(scan: ImageData,
                                             subset_size: int = 30,
                                             low_memory_mode: int = True) -> None:
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
                                 n_erosions: int = 2,
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
            LabelsData: Mask where air is labeled as 1, polymer as 2, and particles as 3.
            """
        if low_memory_mode:
            mask: LabelsData = segment_scan(scan=scan,  # type:ignore
                                            otsu_sigma=otsu_sigma,
                                            particle_mask_sigma=particle_mask_sigma,
                                            particle_enlarge_radius=particle_enlarge_radius,
                                            smooth_labels_radius=smooth_labels_radius,
                                            low_memory_mode=True)
            return mask
        else:
            return segment_scan(scan=scan,  # type:ignore
                                otsu_sigma=otsu_sigma,
                                particle_mask_sigma=particle_mask_sigma,
                                particle_erosions=n_erosions,
                                particle_enlarge_radius=particle_enlarge_radius,
                                smooth_labels_radius=smooth_labels_radius,
                                low_memory_mode=False)

    n_slices, _, _ = scan.shape
    subscan = scan[n_slices // 2 - subset_size // 2:n_slices // 2 + subset_size // 2, :, :]

    viewer = napari.Viewer()
    viewer.add_image(subscan)  # type:ignore
    viewer.window.add_dock_widget(interactive_segmentation, area='right')
