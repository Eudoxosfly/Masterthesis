import numpy as np

import skimage


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
