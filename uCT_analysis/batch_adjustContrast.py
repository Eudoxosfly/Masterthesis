import multiprocessing
from functools import partial
import os
import numpy as np
from skimage import io
from skimage.transform import downscale_local_mean
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage import filters


def get_img(img_path):
    img = io.imread(img_path)
    return img


def get_paths(root):
    paths = []
    for root, dirs, files in os.walk(root):
        for file in files:
            if file.endswith('.tif'):
                paths.append(os.path.join(root, file))
    return paths


def reduce_bit(img):
    img = img_as_ubyte(img)
    return img


def save_img(img, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    io.imsave(path, img)


def pool_image(img, factor=2):
    img = downscale_local_mean(img, factor).astype(np.uint8)
    return img


def adjust_contrast(img, p):
    img_rescale = exposure.rescale_intensity(img, in_range=p)
    return img_rescale


def convert(p, to_png, path):
    if to_png:
        img = get_img(path)
        img = adjust_contrast(img, p)
        img = filters.median(img)
        img = reduce_bit(img)
        img = pool_image(img)
        folder, name = os.path.split(path)
        name = name.split('.')[0]
        save_img(img, os.path.join(folder + "_png", name + '.png'))
        return
    else:
        img = get_img(path)
        img = adjust_contrast(img, p)
        folder, name = os.path.split(path)
        name = name.split('.')[0]
        save_img(img, os.path.join(folder + "_processed", name + '.tif'))


def run(to_png=False):
    PROCESSES = 10
    print('Creating pool with %d processes\n' % PROCESSES)
    path = "E:/Measurements"
    folders = [
        # "AC15_2",
        # "AC14_2",
        # "AC11_2",
        # "AC08",
        # "AC04_2",
        # "AB01_2",
        # "AB01_1",
        # "AD01",
        # "AC14_3",
        # "AD02",
        # "AD07",
        # "AD08",
        # "AD09",
        # "AD10",
        "AD00",
        "AD17",
        "AD06",
        "AD12",
        "AD14",
        "AD15",
        "AD16",
        "AD18",
        "AD19",
        "AD20",
    ]
    folders = [path + "/" + folder + "/" + "Slices" for folder in folders]
    for folder in folders:
        standard_img = get_img(os.path.join(folder, "slice00700.tif"))
        p = tuple(np.percentile(standard_img, (10, 100)))
        this_convert = partial(convert, p, to_png)
        with multiprocessing.Pool(PROCESSES) as pool:
            root = folder
            paths = get_paths(root)
            print("Found {} images in {}.".format(len(paths), folder))
            to_png and print("Converting images to .png, reducing resolution, bit depth and using median sharpening...")
            not to_png and print("Enhancing contrast for images...")
            pool.map(this_convert, paths)
        print("Done with folder {}".format(folder))
        print("-" * 30)


if __name__ == '__main__':
    multiprocessing.freeze_support()
    run(to_png=True)
