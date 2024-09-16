import os

import napari
import numpy as np
import skimage.io
from tqdm import tqdm


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
