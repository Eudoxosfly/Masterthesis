import numpy as np
import skimage

def print_image_info(image):
    print("\t Shape: {}".format(image.shape))
    print("\t Range: {}-{}".format(image.min(), image.max()))
    print("\t Dtype: {}".format(image.dtype))
    print("\t Unique: {}".format(np.unique(image).shape[0]))


# Generate random colormap
def rand_cmap(nlabels: int | None=None, label_image: np.ndarray | None=None):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    Adapted from delestro: https://github.com/delestro/rand_cmap/blob/master/rand_cmap.py
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np

    if nlabels is None and label_image is None or nlabels is not None and label_image is not None:
        raise ValueError("Either nlabels or label_image must be provided, not both, not none.")
    if nlabels is None:
        nlabels = len(np.unique(label_image))


    randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                      np.random.uniform(low=0.2, high=1),
                      np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

    # Convert HSV list to RGB
    randRGBcolors = []
    for HSVcolor in randHSVcolors:
        randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

    random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    return random_colormap

def adjust_contrast(im: np.ndarray[np.uint16],
                    min_percentile: int = 0,
                    max_percentile: int = 100) -> np.ndarray[np.uint16]:
    p1, p2 = np.percentile(im, (min_percentile, max_percentile))
    img_rescale = skimage.exposure.rescale_intensity(im, in_range=(p1, p2))
    return img_rescale

def get_transpose_order(stack: np.ndarray,
                         axis: str = "z") -> list:
    if axis not in ["z", "y", "x"]:
        raise ValueError("Invalid axis. Choose 'z', 'y' or 'x'.")
    dim = stack.shape
    min_idx = int(np.argmin(dim))
    order = [0, 1, 2]
    order.remove(min_idx)
    order.insert({"z": 0, "y": 1, "x": 2}[axis], min_idx)
    return order
