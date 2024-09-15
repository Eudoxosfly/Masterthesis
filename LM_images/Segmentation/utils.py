import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def apply_watershed(img: np.array, opening_iter: int = 2, closing_iter: int = 3, dist_transform_threshold: float = 0.5,
                    output_steps: bool = True) -> tuple[np.array, np.array]:
    gray: np.array = cv.cvtColor(img.copy(), cv.COLOR_BGR2GRAY)
    thresh: np.array = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Determine fore- and background area
    kernel: np.array = np.ones((3, 3), np.uint8)
    opening: np.array = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=opening_iter)
    sure_bg: np.array = cv.dilate(opening, kernel, iterations=closing_iter)

    dist_transform: np.array = cv.distanceTransform(opening, cv.DIST_L2, 5)
    sure_fg: np.array = cv.threshold(dist_transform, dist_transform_threshold * dist_transform.max(), 255, 0)[1]

    # Finding unknown region
    sure_fg: np.array = np.uint8(sure_fg)
    unknown: np.array = cv.subtract(sure_bg, sure_fg)
    # Marker labelling, setting background to one (+1), setting unknown to zero
    markers: np.array = cv.connectedComponents(sure_fg)[1] + 1
    markers[unknown == 255] = 0

    markers_final: np.array = cv.watershed(img, markers.copy())
    img[markers_final == -1] = [255, 0, 0]

    if output_steps:
        height, width, _ = img.shape
        aspect_ratio = height / width
        fig_x = aspect_ratio * 20
        fig_y = aspect_ratio * 40 / 2
        fig, axs = plt.subplots(4, 2, figsize=(fig_x, fig_y))
        axs[0, 0].imshow(gray, cmap="gray")
        axs[0, 1].imshow(thresh, cmap="gray")
        axs[1, 0].imshow(sure_bg, cmap="gray")
        axs[1, 1].imshow(sure_fg, cmap="gray")
        axs[2, 0].imshow(markers, cmap="jet")
        axs[2, 1].imshow(unknown, cmap="gray")
        axs[3, 0].imshow(markers_final, cmap="jet")
        axs[3, 1].imshow(img)

    return markers_final, img


from typing import List


def build_all_layer_point_grids(n_per_side: int, n_layers: int, scale_per_layer: int) -> List[np.ndarray]:
    """Generates point grids for all crop layers."""
    points_by_layer = []
    for i in range(n_layers + 1):
        n_points = int(n_per_side / (scale_per_layer ** i))
        points_by_layer.append(build_point_grid(n_points))
    return points_by_layer


def build_point_grid(n_per_side: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = np.linspace(offset, 1 - offset, n_per_side)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


# better point grid builder
def build_point_grid2(n_x: int, n_y: int) -> np.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    from itertools import product
    offset_x = 1 / (2 * n_x)
    offset_y = 1 / (2 * n_y)
    xx = np.linspace(0 + offset_x, 1 - offset_x, n_x)
    yy = np.linspace(0 + offset_y, 1 - offset_y, n_y)
    points = np.array(list(product(xx, yy)))
    return points
