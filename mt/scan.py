import os.path
import pickle

import pandas as pd
from skimage.transform import downscale_local_mean

from mt.ct_utils import *
from mt.utils import rand_cmap, get_transpose_order
from mt.visualization import *


class Scan:
    """Class for handling uCT scans.

    The Scan class is used to load, process and analyze uCT scans. It provides methods to load the scan and its
    properties from files, segment the scan into a mask and a particle mask, calculate properties of the scan and
    the particles, and visualize the scan and the masks. The class also provides methods to save the scan and its
    properties to files.

    Methods:
        ## IO methods
        load: Load scan and properties from the path.
        save: Save the scan and its properties to files.
        export_image: Save a slice of either scan or mask to a file.

        ## Segmentation methods
        segment: Segment the scan into a mask.
        segment_particles: Segment the scan into a particle mask.
        voronoi_tesselation: Perform a Voronoi tesselation on the particle mask.
        try_segmentation_settings: Try segmentation settings on a subset of the scan.
        set_segmentation_settings: Set the segmentation settings.
        set_particle_segmentation_settings: Set the particle segmentation settings.

        ## Analysis methods
        calculate_properties: Calculate properties of the scan and the particles.
        contact_area_analysis: Calculate the contact area between the particles.
        mask_analysis: Calculate properties of the mask.
        calculate_particle_statistics: Calculate properties of the particles.

        ## Visualization methods
        show: Show the scan and the masks in napari.
        show_nb: Show the scan and the masks in a Jupyter notebook.
        show_hist: Show a histogram of the scan.

        ## Utility methods
        get_stack: Get the scan.
        get_mask: Get the mask.
        get_particle_mask: Get the particle mask.
        get_tesselation: Get the tesselation.

    Attributes:
        path (str): Path to the scan.
        export_path (str): Path to the export folder.
        downscale (bool): Whether to downscale the scan.
        _stack (np.ndarray): Scan for internal use.
        _mask (np.ndarray): Mask for internal use.
        _particle_mask (np.ndarray): Particle mask for internal use.
        _tesselation (np.ndarray): Tesselation for internal use.
        voxel_size_mm (float): Voxel size in mm.
        voxel_volume_mm3 (float): Volume of a voxel in mm^3.
        scan_dimensions_mm (tuple): Dimensions of the scan in mm.
        V_mm3 (float): Volume of the scan in mm^3.
        analytics (dict): Dictionary to store analysis results.
        slice_range (tuple): Range of slices to consider.
        discard_ends (bool): Whether to discard the ends of the scan.
        segmentation_settings (SegmentationSettings): Segmentation settings.
        particle_segmentation_settings (SegmentationSettings): Particle segmentation settings.
        """
    discard_ends_size = 80

    def __init__(self, path: str,
                 discard_ends: bool = True,
                 downscale: bool = False):
        self.path: str = path
        self.export_path: str = self._set_export_path()
        self.downscale: bool = downscale

        self._stack: np.ndarray[np.uint16] | None = None
        self._mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None
        self._particle_mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None
        self._tesselation: np.ndarray[np.uint16] | None = None

        self.voxel_size_mm: float | None = None
        self.voxel_volume_mm3: float | None = None
        self.scan_dimensions_mm: tuple[float, float, float] | None = None
        self.V_mm3: float | None = None
        self.analytics: dict = dict()

        self.slice_range: tuple[int, int] | None = None
        self.discard_ends: bool = discard_ends
        self.segmentation_settings: SegmentationSettings | None = None
        self.particle_segmentation_settings: SegmentationSettings | None = None
        cle.select_device("RTX")

    # %%
    # IO methods
    def load(self,
             refresh: bool = False,
             logging: bool = False) -> None:
        """Load scan and properties from the path.

        Tries to load the scan properties and analysis results from a pickled Scan object. If none can be found,
        the available properties are read from files.

        Args:
            refresh (bool): Whether to not load the pickled Scan object.
            logging (bool): Whether to print logging information.

        Returns:
            None
        """

        # load the stack
        if self._scan_object_exists() and not refresh:
            print("Loading pickled Scan object from: {}".format(self.export_path + "Scan.pkl"))
            self._load_scan_object()
        else:
            self.voxel_size_mm = read_scan_properties(self.path)

        self._load_stack(logging=logging)
        self._np_load("_mask", logging=logging)
        self._np_load("_particle_mask", logging=logging)
        self._np_load("_tesselation", logging=logging)

        if self.downscale: self._downscale_stack()

    def save(self, logging: bool = False):
        """Save the scan and its properties to files.

        Saves the scan and its properties to a pickled Scan object, the segmentation, particle mask and Voronoi
        tesselation to numpy files. """

        all_attributes = {}
        for key, value in self.__dict__.items():
            if key not in ["_stack", "_mask", "_particle_mask", "_tesselation", "path", "export_path", "downscale"]:
                all_attributes[key] = value
        with open(self.export_path + "Scan.pkl", "wb") as f:
            pickle.dump(all_attributes, f)
        logging and print("Saved Scan object to: {}".format(self.export_path + "Scan.pkl"))
        self._save_mask()
        logging and print("Saved mask to: {}".format(self.export_path + "_mask.npy"))
        self._save_particle_mask()
        logging and print("Saved particle mask to: {}".format(self.export_path + "_particle_mask.npy"))
        self._save_tesselation()

    def export_image(self,
                     image_type: str = "stack",
                     axis: str = "z",
                     slice_idx: int = None,
                     region_of_interest: tuple[float, float, float] = None,
                     aspect_ratio: float = 2,
                     fixed_length_type: str = "overview"):
        """Save a slice of either scan or a mask to a file.

        Args:
            image_type (str): Type of image to export. Choose from 'stack', 'mask', 'particle_mask' or 'tesselation'.
            axis (str): Axis to slice along. Choose from 'x', 'y' or 'z'.
            slice_idx (int): Index of the slice to export.
            region_of_interest (tuple): Region of interest to export. Format: (x_min, y_min, width).
            aspect_ratio (float): Aspect ratio of the exported image.
            fixed_length_type (str): Type of fixed length to use. Choose from 'overview', 'particle' or 'detail'.
            """
        if self.downscale:
            print("Stack was downscaled, operation cancelled. Please load the full scan.")
            return
        image_types = {"stack": self.get_stack(),
                       "mask": self.get_mask(),
                       "particle_mask": self.get_particle_mask(),
                       "tesselation": self.get_tesselation()}
        if image_type not in image_types.keys():
            raise ValueError("Invalid image type. Choose 'stack', 'mask', 'particle_mask' or 'tesselation'.")
        stack = image_types[image_type]
        stack = reslice(stack, axis)
        if slice_idx is None:
            slice_idx = stack.shape[0] // 2

        export_image(im=stack[slice_idx],
                     file_path=self.export_path + image_type + "_slice_" + str(slice_idx) + ".png",
                     scale_mm=self.voxel_size_mm,
                    fixed_length_type=fixed_length_type,
                     region_of_interest=region_of_interest,
                     aspect_ratio=aspect_ratio)

    # %%
    # Segmentation methods
    def try_segmentation_settings(self,
                                  subset_size: int = 100,
                                  autorun: bool = True,
                                  segment_particles_only: bool = False):
        """Try segmentation settings on a subset of the scan interactively in napari"""

        adjust_segmentation_parameters_on_subset(scan=self.get_stack(),
                                                 subset_size=subset_size,
                                                 autorun=autorun,
                                                 segment_particles_only=segment_particles_only)

    def set_segmentation_settings(self, settings: SegmentationSettings):
        self.segmentation_settings = settings

    def set_particle_segmentation_settings(self, settings: SegmentationSettings):
        self.particle_segmentation_settings = settings

    def segment(self):
        """Segment the scan into air, polymer and aluminum regions."""
        mask = segment_scan(self.get_stack(),
                            settings=self.segmentation_settings)
        # pad mask with zero images to match the original stack
        self._mask = self._pad_mask(mask)

    def segment_particles(self):
        """Create instance segmentation of the particles."""
        mask = particle_segmentation(self.get_stack(),
                                     settings=self.particle_segmentation_settings)

        self._particle_mask = self._pad_mask(mask)

    def voronoi_tesselation(self):
        """Perform a Voronoi tesselation on the particles."""
        v_tess = voronoi_tesselation(self.get_stack(),
                                     settings=self.particle_segmentation_settings)

        self._tesselation = self._pad_mask(v_tess)

    # %%
    # Analysis methods

    def calculate_properties(self,
                             logging: bool = False):
        """Calculate all properties of the scan and the particles."""
        self.scan_dimensions_mm = self._calc_dimensions()
        self.V_mm3 = self._calc_volume()
        logging and print("Starting mask analysis.")
        self.mask_analysis()
        logging and print("Starting particle analysis.")
        self.calculate_particle_statistics()
        logging and print("Starting contact area analysis.")
        self.contact_area_analysis()

    def contact_area_analysis(self, grid: tuple[int, int] = (5, 5)):
        """Generate analytics for the bonding plane slice."""
        tess = np.transpose(self.get_tesselation(), get_transpose_order(stack=self.get_stack(), axis="z"))
        mask = np.transpose(self.get_mask(), get_transpose_order(stack=self.get_stack(), axis="z"))
        mid_idx = tess.shape[0] // 2
        mask_mid = mask[mid_idx]
        tess_mid = tess[mid_idx]
        cell_area, contact_pct = get_areas_and_contact(tesselation_2d=tess_mid,
                                                       mask_2d=mask_mid,
                                                       voxel_area=self.voxel_size_mm ** 2,
                                                       averaging_method="mean",
                                                       grid=grid)
        self.analytics.update({"2d_cell_area": cell_area, "2d_contact_pct": contact_pct})

    def mask_analysis(self):
        """Generate contact analytics for the mask."""
        props: dict = dict()
        props["air_Al_contact_area_mm2]"] = contact_area(self.get_mask(), 1, 3) * self.voxel_size_mm ** 2
        props["polymer_Al_contact_area_mm2"] = contact_area(self.get_mask(), 2, 3) * self.voxel_size_mm ** 2
        props["contact_air_Al_percent"] = (props["air_Al_contact_area_mm2]"] / (
                props["air_Al_contact_area_mm2]"] + props["polymer_Al_contact_area_mm2"]) * 100)
        props["total_air_volume_mm3"] = self._mask[self._mask == 1].size * self.voxel_size_mm ** 3

        self.analytics.update({"mask_analytics": props})

    def calculate_particle_statistics(self):
        """Calculate statistics of the particles."""
        mask = cle.pull(cle.exclude_labels_on_edges(self.get_particle_mask()))
        props = cle.statistics_of_labelled_pixels(mask)
        volumes_voxel, _ = skimage.exposure.histogram(mask)
        props["volume_mm3"] = volumes_voxel[1:] * self.voxel_size_mm ** 3
        self.analytics.update({"particle_statistics": props})

    # %%
    # Utility methods
    def show(self,
             axis: str = "y"):
        """Utility function to show the scan with all masks overlayed in napari.

        Args:
            axis (str): Axis to show. Choose from 'x', 'y' or 'z'.
        """
        order = get_transpose_order(stack=self.get_stack(), axis=axis)
        t = lambda x: np.transpose(x, order) if x is not None else None
        show_in_napari(t(self.get_stack()),
                       t(self.get_mask()),
                       t(self.get_particle_mask()),
                       t(self.get_tesselation())
                       )

    def show_nb(self,
                mask_type: str = "mask",
                alpha=0.3,
                x_size: int = 30,
                axis: str = "z") -> None:
        """Utility function to show the scan with a mask overlayed in a Jupyter notebook
        when napari is not available.

        Args:
            mask_type (str): Type of mask to show. Choose from 'mask', 'particle_mask' or 'tesselation'.
            alpha (float): Transparency of the mask overlay.
            x_size (int): Width modifier for the figure.
            axis (str): Axis to show. Choose from 'y' or 'z'.
        """
        mask_types: dict = {"mask": self.get_mask(),
                            "particle_mask": self.get_particle_mask(),
                            "tesselation": self.get_tesselation()}
        if mask_type not in mask_types.keys():
            raise ValueError("Invalid mask type. Choose 'mask', 'particle_mask' or 'tesselation'.")
        segmentation = mask_types[mask_type]
        if axis == "y":
            order = get_transpose_order(stack=self.get_stack(), axis="y")
            mask = np.transpose(segmentation, order)
            im = np.transpose(self.get_stack(), order)
            y_size = 3 * im.shape[1] / im.shape[2] * x_size + 1
            fig, axs = plt.subplots(3, 1, figsize=(x_size, y_size))
            axs[0].imshow(im[0], cmap="gray")
            axs[0].set_title("First")
            axs[0].imshow(mask[0],
                          cmap=rand_cmap(label_image=mask[0])
                          if mask_type != "mask" else "hot",
                          alpha=alpha)
            axs[0].axis("off")

            axs[1].imshow(im[self.__len__() // 2], cmap="gray")
            axs[1].set_title("Middle")
            axs[1].imshow(mask[self.__len__() // 2],
                          cmap=rand_cmap(label_image=mask[self.__len__() // 2])
                          if mask_type != "mask" else "hot",
                          alpha=alpha)

            axs[2].imshow(im[-1], cmap="gray")
            axs[2].set_title("Last")
            axs[2].imshow(mask[-1],
                          cmap=rand_cmap(label_image=mask[-1])
                          if mask_type != "mask" else "hot",
                          alpha=alpha)
        elif axis == "z":
            order = get_transpose_order(stack=self.get_stack(), axis="z")
            mask = np.transpose(segmentation, order)
            im = np.transpose(self.get_stack(), order)
            n, h, w = im.shape
            fig, axs = plt.subplots(1, figsize=(x_size, x_size))
            axs.imshow(im[n // 2], cmap="gray")
            axs.imshow(mask[n // 2],
                       cmap=rand_cmap(label_image=segmentation[0])
                       if mask_type != "mask" else "prism",
                       alpha=alpha)
            axs.axis("off")

    def show_hist(self):
        """Show a histogram of the scan.
        """
        fig, axs = plt.subplots(1, figsize=(20, 10))
        axs.hist(self.get_stack().flatten(), bins=200)
        # logarithmic axis
        axs.set_yscale("log")
        # grid
        axs.grid(True)

    ## Getter, setter and helper methods
    def get_stack(self):
        if self._stack_exists():
            return self._apply_slice(self._stack)

    def get_mask(self):
        if self._mask_exists():
            return self._apply_slice(self._mask)

    def get_particle_mask(self):
        if self._particle_mask_exists():
            return self._apply_slice(self._particle_mask)

    def get_tesselation(self):
        if self._tesselation_exists():
            return self._apply_slice(self._tesselation)

    # %%
    ## Private methods for internal use
    #
    # segmentation utility methods
    def _apply_slice(self, stack):
        """Apply slice range or discard ends to the stack."""
        if self.slice_range is not None:
            return stack[self.slice_range[0]:self.slice_range[1], :, :]
        elif self.discard_ends:
            return stack[self.discard_ends_size:-self.discard_ends_size, :, :]
        else:
            return stack

    def _pad_mask(self, mask):
        """Pad the mask with zeros to match the original stack."""
        if self.slice_range is not None:
            return np.pad(mask,
                          ((self.slice_range[0], self.get_stack().shape[0] - self.slice_range[1]), (0, 0), (0, 0)))
        elif self.discard_ends:
            return np.pad(mask, ((self.discard_ends_size, self.discard_ends_size), (0, 0), (0, 0)))
        else:
            return mask

    # calculation methods
    def _calc_dimensions(self):
        """Calculate the dimensions of the scan in mm."""
        h, w, d = self.get_stack().shape
        return (h * self.voxel_size_mm,
                w * self.voxel_size_mm,
                d * self.voxel_size_mm)

    def _calc_volume(self) -> float:
        """Calculate the volume of the scan in mm^3."""
        return float(np.prod(self.scan_dimensions_mm))

    # IO methods
    def _set_export_path(self):
        """Sets the export path for the scan."""
        ex_path = self.path.replace("04_uCT", "06_Results/uCT")
        if not os.path.exists(ex_path):
            os.makedirs(ex_path)
        return ex_path

    def _np_load(self, name, logging: bool = False):
        """Load stack from numpy file."""
        if not os.path.exists(self.export_path + name + ".npy"):
            logging and print("No {} file found at: {}".format(name, self.export_path + name + ".npy"))
            return

        setattr(self, name, np.load(self.export_path + name + ".npy"))

        logging and print("Loaded {} from: {}".format(name, self.export_path + name + ".npy"))

    def _load_stack(self, logging: bool = False):
        """Load stack from image files."""
        self._stack = load_stack(path=self.path,
                                 folder="Slices",
                                 logging=logging)

    def _load_scan_object(self):
        """Load the scan object from a pickled file."""
        with open(self.export_path + "Scan.pkl", "rb") as f:
            all_attributes = pickle.load(f)
            for key, value in all_attributes.items():
                if key not in ["path", "downscale", "export_path"]:
                    setattr(self, key, value)

    def _save_mask(self):
        if self._mask_exists():
            np.save(self.export_path + "_mask.npy", self._mask)

    def _save_particle_mask(self):
        if self._particle_mask_exists():
            np.save(self.export_path + "_particle_mask.npy", self._particle_mask)

    def _save_tesselation(self):
        if self._tesselation_exists():
            np.save(self.export_path + "_tesselation.npy", self._tesselation)

    def _downscale_stack(self):
        """Downscale the stack by a factor of 2.

        The stack is binned by a factor of 2 in each dimension. The image is first cropped to an even size to allow
        for binning without artifacts."""
        me = lambda x: x if x % 2 == 0 else x - 1
        n, w, h = self._stack.shape
        self._stack = downscale_local_mean(self._stack[:me(n), :me(w), :me(h)],
                                           (2, 2, 2)).astype(np.uint16)
        print("Downscaled stack to: {}".format(self._stack.shape))

    # Methods to check existence of attributes
    def _stack_exists(self):
        return self._stack is not None

    def _mask_exists(self):
        return self._mask is not None

    def _particle_mask_exists(self):
        return self._particle_mask is not None

    def _tesselation_exists(self):
        return self._tesselation is not None

    def _scan_object_exists(self):
        return os.path.exists(self.export_path + "Scan.pkl")

    def __getitem__(self, item):
        """Define slice behavior for the Scan object."""
        return self.get_stack()[item]

    def __str__(self):
        return (
                "Scan with shape {} and voxel size {:.2f}."
                .format(self.get_stack().shape, self.voxel_size_mm) +
                "\n Scanned volume: ({:.1f}x{:.1f}x{:.1f}) mm".format(*self.scan_dimensions_mm) +
                "\n Volume: {:.2f} mm^3".format(self.V_mm3)
        )

    def __len__(self):
        return self.get_stack().shape[0]
