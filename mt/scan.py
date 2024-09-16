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

@dataclass(kw_only=True)
class Scan:
    scan: np.ndarray
    voxel_size_um: float
    scan_dimensions_mm: tuple[float, float, float] = None
    V: float = None

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
        return (h * self.voxel_size_um * 1e3,
                w * self.voxel_size_um * 1e3,
                d * self.voxel_size_um * 1e3)

    def calc_volume(self) -> float:
        return float(np.prod(self.scan_dimensions_mm))

    def reslice(self,
                axis: tuple[int, int, int] = (1, 0, 2)) -> np.ndarray:
        self.scan = np.transpose(self.scan, axis)


def load_scan(path: str,
              ending: str = ".tif",
              logging: bool = False,
              image_range: tuple | None = None):

    def read_properties(path: str):
        with open(path + "unireconstruction.xml", "r") as f:
            text = f.read()
        voxel_size = float(text.split("<voxelSize X=\"")[1].split("\"")[0])
        return voxel_size

    def load_stack(path: str,
                   ending: str,
                   image_range: tuple | None,
                   logging: bool):
        path = path + "Slices/"
        start = time.time()
        logging and print("Loading images from: ", path)
        files = [path + file for file in os.listdir(path) if file.endswith(ending)]
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

    scan = load_stack(path=path,
                      ending=ending,
                      logging=logging,
                      image_range=image_range)
    voxel_size = read_properties(path)

    scan = Scan(scan=scan,
                voxel_size_um=voxel_size)
    scan.calculate_properties()
    return scan


def particle_segmentation(im: ImageData,
                          sigma: float = 0.1,
                          n_erosions: int = 2,
                          dilation_radius: int = 1) -> LabelsData:
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


def segment_scan(scan: ImageData,
                 logging: bool = True,
                 otsu_sigma: float = 1,
                 particle_enlarge_radius: int = 1,
                 particle_mask_sigma: float = 0.1,
                 particle_erosions: int = 2,
                 smooth_labels_radius: int = 0) -> LabelsData:
    def _particle_segmentation(im: ImageData,
                               sigma: float = 0.1,
                               dilation_radius: int = 1) -> LabelsData:
        # allocate memory on the gpu
        smoothed_gpu = cle.create_like(im, dtype=np.float32)
        mask_gpu = cle.create_like(im, dtype=np.uint32)

        # apply gaussian blur and otsu threshold
        cle.gaussian_blur(im, output_image=smoothed_gpu, sigma_x=sigma, sigma_y=sigma, sigma_z=sigma)
        cle.threshold_otsu(smoothed_gpu, output_image=mask_gpu)
        del smoothed_gpu

        # morphological operations removed for performance reasons
        cle.dilate_labels(mask_gpu, output_image=mask_gpu, radius=dilation_radius)
        cle.erode_connected_labels(mask_gpu, output_image=mask_gpu, radius=1)

        mask = cle.pull(mask_gpu)
        del mask_gpu

        return mask

    scan_gpu = cle.push(scan)
    smoothed_gpu = cle.create_like(scan_gpu, dtype=np.float32)
    cle.gaussian_blur(scan_gpu, output_image=smoothed_gpu, sigma_x=otsu_sigma, sigma_y=otsu_sigma, sigma_z=otsu_sigma)
    im = cle.pull(smoothed_gpu)
    del smoothed_gpu
    logging and print("Smoothed image.")
    th1, th2 = threshold_multiotsu(im, classes=3)
    otsu_air_mask = im <= th1
    otsu_polymer_mask = (im > th1) & (im < th2)
    otsu_particle_mask = im >= th2
    logging and print("Finished otsu masks.")
    fine_particle_mask = _particle_segmentation(scan_gpu,
                                                sigma=particle_mask_sigma,
                                                dilation_radius=particle_enlarge_radius)

    logging and print("Finished refined particle mask.")
    # combine them
    mask = np.zeros_like(im, dtype=np.uint32)
    mask[otsu_air_mask] = 1
    mask[otsu_polymer_mask] = 2
    mask[otsu_particle_mask] = 3
    mask[fine_particle_mask > 0] = 3

    logging and print("Created full segmentation.")

    if smooth_labels_radius > 0:
        mask_gpu = cle.push(mask)
        cle.smooth_labels(mask_gpu, output_image=mask_gpu, radius=smooth_labels_radius)
        mask = cle.pull(mask_gpu)
        del mask_gpu

    return mask


def process_subset_in_napari(scan: ImageData, subset_size: int = 30):
    @magicgui(auto_call=True)
    def segment(scan: ImageData,
                logging: bool = False,
                otsu_sigma: float = 0.6,
                particle_enlarge_radius: int = 1,
                particle_mask_sigma: float = 0.1,
                smooth_labels_radius: int = 2) -> LabelsData:
        return segment_scan(scan=scan,
                            logging=logging,
                            otsu_sigma=otsu_sigma,
                            particle_enlarge_radius=particle_enlarge_radius,
                            particle_mask_sigma=particle_mask_sigma,
                            smooth_labels_radius=smooth_labels_radius)

    n_slices, _, _ = scan.shape
    subscan = scan[n_slices // 2 - subset_size // 2:n_slices // 2 + subset_size // 2, :, :]

    viewer = napari.Viewer()
    viewer.add_image(subscan)
    viewer.window.add_dock_widget(segment, area='right')
