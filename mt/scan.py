import os.path
import pickle

import pandas as pd

from mt.ct_utils import *


class Scan:
    """Class for handling uCT scans.

    Methods:
        IO:
        load: Load the scan and properties from the path.
        save: Save the Scan object to a pickle file.
        save_mask: Save the mask to the path.

        Segmentation:
        try_segmentation_settings: Try different segmentation settings interactively on a subset of the scan.
        set_segmentation_settings: Set the segmentation settings.
        segment: Segment the scan.

        Analysis:
        calculate_properties: Group method to calculate all properties.
        calculate_volume: Calculate the volume of the scan.
        calculate_dimensions: Calculate the dimensions of the scan.
        calculate_contact_area: Calculate the contact area of the mask labels.
        calculate_particle_statistics: Calculate statistics of the particles.


        Utility:
        show: Show the scan.
        reslice: Reslice the scan.



    Attributes:
        path (str): Path to the scan.
        stack (np.ndarray): 3D numpy array with the scan with slice_range or discard_ends applied.
        _stack (np.ndarray): 3D numpy array with the original scan.
        mask (np.ndarray): 3D numpy array with the mask with slice_range or discard_ends applied.
        _mask (np.ndarray): 3D numpy array with the original mask.

        voxel_size_mm (float): Voxel size in mm.
        scan_dimensions_mm (tuple): Dimensions of the scan in mm.
        V_mm3 (float): Volume of the scan in mm^3.
        contact_areas_mm2 (dict): Contact areas of the mask labels in mm^2.
        particle_statistics (dict): Statistics of the particles.
        n_particles (int): Number of particles.

        slice_orientation (list): Orientation of the slices.
        slice_range (tuple): Range of slices used for analysis
        discard_ends (bool): Whether to discard the ends of the scan.
        segmentation_settings (SegmentationSettings): Settings for the segmentation.
        """

    def __init__(self, path: str,
                 discard_ends: bool = True):
        self.path: str = path

        self.stack: np.ndarray[np.uint16] | None = None
        self._stack: np.ndarray[np.uint16] | None = None
        self.mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None
        self._mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None
        self.particle_mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None
        self._particle_mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None

        self.voxel_size_mm: float | None = None
        self.voxel_volume_mm3: float | None = None
        self.scan_dimensions_mm: tuple[float, float, float] | None = None
        self.V_mm3: float | None = None
        self.mask_analytics: pd.DataFrame | None = None
        self.particle_statistics: pd.DataFrame | None = None

        self.slice_orientation: str = "y"
        self.slice_range: tuple[int, int] | None = None
        self.discard_ends: bool = discard_ends
        self.segmentation_settings: SegmentationSettings = SegmentationSettings()
        self.particle_segmentation_settings: SegmentationSettings = SegmentationSettings()

    # %%
    # IO methods
    def load(self,
             refresh: bool = False,
             logging: bool = False) -> None:
        """Load scan and properties from the path.

        Tries to load the scan properties and analysis results from a pickled Scan object. If none can be found,
        the available properties are read from files.

        Args:
            logging (bool): Whether to print logging information.

        Returns:
            None
        """

        # load the stack
        if os.path.exists(self.path + "Scan.pkl") and not refresh:
            print("Loading pickled Scan object from: {}".format(self.path + "Scan.pkl"))
            self._load_scan_object()
        else:
            # TODO: read other relevant properties from the files
            self.voxel_size_mm = read_scan_properties(self.path)

        self._load_stack(logging=logging)
        self._load_mask(logging=logging)
        self._load_particle_mask(logging=logging)

    def save(self):
        all_attributes = {}
        for key, value in self.__dict__.items():
            if key not in ["stack", "mask", "_stack", "_mask", "particle_mask", "_particle_mask"]:
                all_attributes[key] = value
        with open(self.path + "Scan.pkl", "wb") as f:
            pickle.dump(all_attributes, f)

        self._save_segmentation()
        self._save_particle_mask()
        self._save_volumes()

    def _save_segmentation(self):
        np.save(self.path + "segmentation.npy", self._mask)

    def _save_particle_mask(self):
        np.save(self.path + "particle_mask.npy", self._particle_mask)

    def _save_volumes(self):
        np.savetxt(self.path + "volumes.csv", self.particle_statistics["volume_mm3"], delimiter="\n")

    # %%
    # Segmentation methods
    def try_segmentation_settings(self,
                                  subset_size: int = 200,
                                  autorun: bool = True,
                                  segment_particles_only: bool = False):

        adjust_segmentation_parameters_on_subset(scan=self.stack,
                                                 subset_size=subset_size,
                                                 autorun=autorun,
                                                 segment_particles_only=segment_particles_only)

    def set_segmentation_settings(self, settings: SegmentationSettings):
        self.segmentation_settings = settings

    def set_particle_segmentation_settings(self, settings: SegmentationSettings):
        self.particle_segmentation_settings = settings

    def segment(self):
        mask = segment_scan(self.stack, settings=self.segmentation_settings)

        # pad mask with zero images to match the original stack
        if self.slice_range is not None:
            self._mask = np.pad(mask,
                                ((self.slice_range[0], self.stack.shape[0] - self.slice_range[1]), (0, 0), (0, 0)))
        elif self.discard_ends:
            self._mask = np.pad(mask, ((80, 80), (0, 0), (0, 0)))

        self.mask = mask

    def segment_particles(self):
        mask = particle_segmentation(self.stack, settings=self.particle_segmentation_settings)

        if self.slice_range is not None:
            self._particle_mask = np.pad(mask, (
                (self.slice_range[0], self.stack.shape[0] - self.slice_range[1]), (0, 0), (0, 0)))
        elif self.discard_ends:
            self._particle_mask = np.pad(mask, ((80, 80), (0, 0), (0, 0)))
        else:
            self._particle_mask = mask
        self.particle_mask = mask

    # %%
    # Analysis methods

    def calculate_properties(self,
                             logging: bool = False):
        self.scan_dimensions_mm = self._calc_dimensions()
        self.V_mm3 = self._calc_volume()
        logging and print("Starting mask analysis.")
        self.mask_analysis()
        logging and print("Starting particle analysis.")
        self.calculate_particle_statistics()

    def mask_analysis(self):
        props: dict = dict()
        props["air_Al_contact_area_mm2]"] = contact_area(self.mask, 1, 3) * self.voxel_size_mm ** 2
        props["polymer_Al_contact_area_mm2"] = contact_area(self.mask, 2, 3) * self.voxel_size_mm ** 2
        props["contact_air_Al_percent"] = (props["air_Al_contact_area_mm2]"] / (
                    props["air_Al_contact_area_mm2]"] + props["polymer_Al_contact_area_mm2"]) * 100)
        props["total_air_volume_mm3"] = self._mask[self._mask == 1].size * self.voxel_size_mm ** 3

        self.mask_analytics = pd.DataFrame(props, index=[0])

    def calculate_particle_statistics(self):
        mask = cle.pull(cle.exclude_labels_on_edges(self.particle_mask))
        props = cle.statistics_of_labelled_pixels(mask)
        volumes_voxel, _ = skimage.exposure.histogram(mask)
        props["volume_mm3"] = volumes_voxel[1:] * self.voxel_size_mm ** 3
        self.particle_statistics = pd.DataFrame(props)

    # %%
    # Utility methods
    def reslice(self):
        self.stack = np.transpose(self.stack, (1, 0, 2))
        self._stack = np.transpose(self._stack, (1, 0, 2))
        if self.mask is not None:
            self.mask = np.transpose(self.mask, (1, 0, 2))
            self._mask = np.transpose(self._mask, (1, 0, 2))
        if self.particle_mask is not None:
            self.particle_mask = np.transpose(self.particle_mask, (1, 0, 2))
            self._particle_mask = np.transpose(self._particle_mask, (1, 0, 2))


    def show(self,
             particle_mask_only: bool = False):
        viewer = napari.Viewer()
        viewer.add_image(self.stack, name="Scan")
        if particle_mask_only:
            if self.particle_mask is not None:
                viewer.add_labels(self.particle_mask, name="Particle mask")
            else:
                print("No particle mask found.")
        else:
            if self.mask is not None:
                viewer.add_labels(self.mask, name="Mask")
            else:
                print("No mask found.")


    def _calc_dimensions(self):
        h, w, d = self.stack.shape
        return (h * self.voxel_size_mm,
                w * self.voxel_size_mm,
                d * self.voxel_size_mm)

    def _calc_volume(self) -> float:
        return float(np.prod(self.scan_dimensions_mm))

    def _load_mask(self, logging: bool = False):
        if not os.path.exists(self.path + "segmentation.npy"):
            logging and print("No segmentation.npy file found at: {}".format(self.path + "segmentation.npy"))
            return

        self._mask = np.load(self.path + "segmentation.npy")

        if self.slice_range is not None:
            self.mask = self._mask[self.slice_range[0]:self.slice_range[1], :, :]
        elif self.discard_ends:
            self.mask = self._mask[80:-80, :, :]
        else:
            self.mask = self._mask

        logging and print("Loaded mask from: {}".format(self.path + "segmentation.npy"))

    def _load_particle_mask(self, logging: bool = False):
        if not os.path.exists(self.path + "particle_mask.npy"):
            logging and print("No particle_mask.npy file found at: {}".format(self.path + "particle_mask.npy"))
            return

        self._particle_mask = np.load(self.path + "particle_mask.npy")

        if self.slice_range is not None:
            self.particle_mask = self._particle_mask[self.slice_range[0]:self.slice_range[1], :, :]
        elif self.discard_ends:
            self.particle_mask = self._particle_mask[80:-80, :, :]
        else:
            self.particle_mask = self._particle_mask

        logging and print("Loaded particle mask from: {}".format(self.path + "particle_mask.npy"))

    def _load_stack(self, logging: bool = False):
        self._stack = load_stack(path=self.path,
                                 folder="Slices",
                                 logging=logging)

        if self.slice_range is not None:
            self.stack = self._stack[self.slice_range[0]:self.slice_range[1], :, :]
        elif self.discard_ends:
            self.stack = self._stack[80:-80, :, :]
        else:
            self.stack = self._stack

    def _load_scan_object(self):
        with open(self.path + "Scan.pkl", "rb") as f:
            all_attributes = pickle.load(f)
            for key, value in all_attributes.items():
                setattr(self, key, value)

    def __getitem__(self, item):
        return self.stack[item]

    def __str__(self):
        return (
                "Scan with shape {} and voxel size {:.2f}."
                .format(self.stack.shape, self.voxel_size_mm) +
                "\n Scanned volume: ({:.1f}x{:.1f}x{:.1f}) mm".format(*self.scan_dimensions_mm) +
                "\n Volume: {:.2f} mm^3".format(self.V_mm3)
        )
