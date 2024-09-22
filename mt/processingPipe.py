from dataclasses import dataclass
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
import scipy


def mean_blur(im: np.ndarray, params: dict[str, any]) -> np.ndarray:
    return cv2.blur(src=im, dst=None, **params)


def gaussian_blur(im: np.ndarray, params: dict[str, any]) -> np.ndarray:
    return cv2.GaussianBlur(src=im, dst=None, **params)


def median_blur(im: np.ndarray, params: dict[str, any]) -> np.ndarray:
    return cv2.medianBlur(src=im, dst=None, **params)


def nlm_blur(im: np.ndarray, params: dict[str, any]) -> np.ndarray:
    nlm = cv2.fastNlMeansDenoising(src=im, dst=None, **params)
    return nlm


def to_8bit(im: np.ndarray, params: dict[str, any] = {}) -> np.ndarray:
    return skimage.util.img_as_ubyte(im)


def downscale_2x(im: np.ndarray, params: dict[str, any] = {}) -> np.ndarray:

    width, height = im.shape
    if width % 2 != 0:
        im = im[:-1, :]
    if height % 2 != 0:
        im = im[:, :-1]

    in_type = im.dtype

    downscaled = skimage.transform.downscale_local_mean(im, (2, 2))
    return downscaled.astype(in_type)


def upscale_2x(im: np.ndarray, params: dict[str, any] = {}) -> np.ndarray:
    img = scipy.ndimage.zoom(im, 2, order=0)
    return img


def threshold_between(im: np.ndarray, params: dict[str, any]) -> np.ndarray:
    mask = (im > params["lower"]) & (im < params["upper"]).copy()
    im[~mask] = 0
    im[mask] = 255
    return im



class ImageProcessingPipe:
    @dataclass
    class Result:
        name: str
        args: dict[str, any] | None
        image: np.ndarray

    def __init__(self, demo_mode=True, stop_at_step=None):
        self.demo_mode = demo_mode
        self.pipe = None
        self.results: dict[int, ImageProcessingPipe.Result] = {}
        self.stop_at_step = stop_at_step

    def create(self, functions: list[tuple[str, dict[str, any]]]):
        self.pipe = functions

    def run(self, image: np.ndarray | str):
        if self.pipe is None:
            raise ValueError("Create a pipeline containing functions before running it")
        self.clear_results()

        if isinstance(image, str):
            image = skimage.io.imread(image)
        self.add_result("Original", image, None)

        for i, (fun, args) in enumerate(self.pipe):
            if self.stop_at_step is not None and i == self.stop_at_step:
                break
            curr_image = self.results[len(self.results)].image.copy()
            image = self.apply_fn(fun, curr_image, args)
            self.add_result(fun, image, args)

    def display_result_list(self):
        for (step, result) in self.results.items():
            print(f"Step {step}: {result.name}:")
            print("\t Shape: {}".format(result.image.shape))
            print("\t Range: {}-{}".format(result.image.min(), result.image.max()))
            print("\t Dtype: {}".format(result.image.dtype))
            print("\t Unique: {}".format(np.unique(result.image).shape[0]))

    def clear_results(self):
        self.results = {}

    def add_result(self, name: str, image: np.ndarray, args: dict[str, any] | None):
        self.results[len(self.results) + 1] = ImageProcessingPipe.Result(name, args, image)

    def show(self, hist=False):
        n = len(self.results)
        if not hist:
            fig, axs = plt.subplots(n, figsize=(20, n * 2))
            for (step, result), ax in zip(self.results.items(), axs):
                ax.imshow(result.image, cmap='gray')
                if result.args is not None:
                    ax.set_title(result.name + "\n" + ", ".join([f"{k}: {v}" for k, v in result.args.items()]))
                else:
                    ax.set_title(result.name)
        else:
            fig, axs = plt.subplots(n, 2, figsize=(20, n * 2))
            for (step, result), (ax1, ax2) in zip(self.results.items(), axs):
                ax1.imshow(result.image, cmap='gray')
                if result.args is not None:
                    ax1.set_title(result.name + "\n" + ", ".join([f"{k}: {v}" for k, v in result.args.items()]))
                else:
                    ax1.set_title(result.name)
                range = (0, 255) if result.image.dtype == np.uint8 else (0, 2**16)
                ax2.hist(result.image.flatten(), bins=256, range=range, density=True)
        fig.tight_layout()
        plt.show()

    def export(self, step):
        image = self.results[step].image
        plt.imsave(f"step_{step}.png", image, cmap='gray')

    def apply_fn(self, fn_name, im: np.ndarray, params: dict[str, any]) -> np.ndarray:
        fns = {
            "nlm_blur": nlm_blur,
            "median_blur": median_blur,
            "mean_blur": mean_blur,
            "gaussian_blur": gaussian_blur,
            "downscale_2x": downscale_2x,
            "to_8bit": to_8bit,
            "threshold_between": threshold_between,
            "upscale_2x": upscale_2x}
        fn = fns.get(fn_name, None)
        if fn is None:
            raise NotImplementedError(f"Function {fn_name} not yet implemented")
        return fn(im, params)
