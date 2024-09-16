import os
import time
from dataclasses import dataclass

import cv2 as cv
import napari
import numpy as np
import pyclesperanto as cle
import skimage
from napari.types import ImageData, LabelsData
from skimage.filters import threshold_multiotsu
from tqdm import tqdm
from magicgui import magicgui


def get_func_gen_settings(on_time, off_time, passes, focus_distance, legacy=False, verbose=False):
    spot_width, spot_length = get_spot_size(focus_distance)

    speed = np.round(spot_length / (on_time + off_time) / passes, 1)
    if legacy:
        speed = spot_length / (on_time + off_time) // passes
        spot_width = 2.8
    total_time_over_spot = spot_length / speed
    frequency = passes / total_time_over_spot
    duty_cycle = on_time / (on_time + off_time)
    total_time_on = total_time_over_spot * duty_cycle
    single_time_on = total_time_on / passes
    total_time_off = total_time_over_spot * (1 - duty_cycle)
    single_time_off = total_time_off / passes

    equivalent_energy = total_time_on * 30 / (spot_width * spot_length)

    settings = {"speed": speed, "frequency": frequency, "duty_cycle": duty_cycle,
                "total_time_over_spot": total_time_over_spot, "passes": passes, "set_on_time": on_time,
                "actual_on_time": single_time_on, "total_on_time": total_time_on, "set_off_time": off_time,
                "actual_off_time": single_time_off, "total_off_time": total_time_off,
                "equivalent_energy": equivalent_energy, }

    if verbose:
        legacy and print("WARNING: Legacy mode is on. Speed is rounded down to the nearest integer.")
        print("Settings:")
        print("Set on time: {:.0f} ms".format(on_time * 1e3))
        print("Set off time: {:.0f} ms".format(off_time * 1e3))
        print("Cycles: {}".format(passes))
        print("-" * 20)
        print("Speed: {} mm/s".format(speed))
        print("Frequency: {:.2f} Hz".format(frequency))
        print("Duty cycle: {:.0f} %".format(duty_cycle * 100))
        print("Total time over spot: {:.2f} s".format(total_time_over_spot))
        print("Equivalent energy: {:.2f} J/mm^2".format(equivalent_energy))
        print("-" * 20)
        print("Actual on time: {:.0f} ms".format(single_time_on * 1e3))
        print("Total on time: {:.0f} ms".format(total_time_on * 1e3))
        print("Actual off time: {:.0f} ms".format(single_time_off * 1e3))
        print("Total off time: {:.0f} ms".format(total_time_off * 1e3))
    return settings


def get_spot_size(distance):
    """Calculate the spot size (mm) in x and y direction for a given focus distance."""

    def get_size(coef, dist):
        return coef[0] * dist + coef[1]

    # coefficients from linear fitting
    x_coef = (0.0498, -0.294)
    y_coef = (0.0599, -0.285)

    x = np.round(get_size(x_coef, distance), 2)
    y = np.round(get_size(y_coef, distance), 2)

    return x, y


def downscale_1024(folder):
    """Downscales a folders images to 1024x1024 using Lanczos interpolation."""

    def down_scale_lanczos(image_path):
        image = skimage.io.imread(image_path)
        image = image[:, :image.shape[0]]
        image = cv.resize(image, (1024, 1024), interpolation=cv.INTER_LANCZOS4)
        return image

    save_path = folder.replace("LM", "LM_downscaled")
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    im_names = [folder + file for file in os.listdir(folder)]
    for name in tqdm(im_names):
        im = down_scale_lanczos(name)
        im = cv.medianBlur(im, 5)
        skimage.io.imsave(name.replace("LM", "LM_downscaled").replace(".tif", ".png"), im)


def load_scan(path: str,
              ending: str = ".tif",
              logging: bool = False,
              image_range: tuple | None = None):
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


def print_image_info(image):
    print("\t Shape: {}".format(image.shape))
    print("\t Range: {}-{}".format(image.min(), image.max()))
    print("\t Dtype: {}".format(image.dtype))
    print("\t Unique: {}".format(np.unique(image).shape[0]))


def contact_area(image, label1, label2):
    """Counts the number of surfaces of label1 in contact with label2.

    Args:
        image(numpy.ndarray): Array representing image (or image stack).
        label1(int): Value of label 1
        label1(int): Value of label 2

    Returns:
        int: Number of voxels of label1 in contact with label2
    """

    histogram = skimage.exposure.histogram(image)

    if label1 not in histogram[1] or label2 not in histogram[1]:
        raise ValueError('One or more labels do not exist. Please input valid labels.')

    x_contact_1 = np.logical_and(image[:, :, :-1] == label1, image[:, :, 1:] == label2)
    x_contact_2 = np.logical_and(image[:, :, :-1] == label2, image[:, :, 1:] == label1)
    y_contact_1 = np.logical_and(image[:, :-1, :] == label1, image[:, 1:, :] == label2)
    y_contact_2 = np.logical_and(image[:, :-1, :] == label2, image[:, 1:, :] == label1)
    z_contact_1 = np.logical_and(image[:-1, :, :] == label1, image[1:, :, :] == label2)
    z_contact_2 = np.logical_and(image[:-1, :, :] == label2, image[1:, :, :] == label1)
    # np.argwhere(hpairs) - counts each pair which is in contact

    contact_voxels = np.count_nonzero(x_contact_1) + np.count_nonzero(x_contact_2) + np.count_nonzero(
        y_contact_1) + np.count_nonzero(y_contact_2) + np.count_nonzero(z_contact_1) + np.count_nonzero(z_contact_2)

    return contact_voxels


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


def reslice(scan: np.ndarray,
            axis: tuple[int, int, int] = (1, 0, 2)) -> np.ndarray:
    return np.transpose(scan, axis)


def show_in_napari(img, *labels):
    viewer = napari.Viewer()
    viewer.add_image(img)
    for label in labels:
        viewer.add_labels(label)


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
    subscan = scan[n_slices//2-subset_size//2:n_slices//2+subset_size//2, :, :]

    viewer = napari.Viewer()
    viewer.add_image(subscan)
    viewer.window.add_dock_widget(segment, area='right')
