import os

import cv2 as cv
import skimage
from tqdm import tqdm


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
