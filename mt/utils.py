import numpy as np

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
