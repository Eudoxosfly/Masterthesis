import numpy as np


def print_image_info(image):
    print("\t Shape: {}".format(image.shape))
    print("\t Range: {}-{}".format(image.min(), image.max()))
    print("\t Dtype: {}".format(image.dtype))
    print("\t Unique: {}".format(np.unique(image).shape[0]))
