import os.path
import pickle

from mt.ct_utils import *
import napari


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
        stack (np.ndarray): 3D numpy array with the scan.
        mask (np.ndarray): 3D numpy array with the mask.

        voxel_size_mm (float): Voxel size in mm.
        scan_dimensions_mm (tuple): Dimensions of the scan in mm.
        V_mm3 (float): Volume of the scan in mm^3.
        contact_areas_mm2 (dict): Contact areas of the mask labels in mm^2.
        particle_statistics (dict): Statistics of the particles.
        n_particles (int): Number of particles.

        slice_orientation (list): Orientation of the slices.
        slice_range (tuple): Range of slices used for analysis
        discard_ends (bool): Whether to discard the ends of the scan.
        low_memory_mode (bool): Whether to reduce the computational expense to run on less memory.
        low_memory_amount (float): Amount of memory to use in low memory mode in GB.
        segmentation_settings (SegmentationSettings): Settings for the segmentation.
        """

    def __init__(self, path: str,
                 low_memory_mode: bool = False,
                 low_memory_amount: float = 1.0,
                 discard_ends: bool = True):
        self.path: str = path

        self.stack: np.ndarray[np.uint16] | None = None
        self.mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None

        self.voxel_size_mm: float | None = None
        self.scan_dimensions_mm: tuple[float, float, float] | None = None
        self.V_mm3: float | None = None

        self.slice_orientation: list = ["y", "z", "x"]
        self.slice_range: tuple[int, int] | None = None
        self.discard_ends: bool = discard_ends
        self.segmentation_settings: SegmentationSettings = SegmentationSettings()
        self.low_memory_mode: bool = low_memory_mode
        self.low_memory_amount: float = low_memory_amount

    #%%
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
        self._load_masks(logging=logging)

    def save(self):
        all_attributes = {}
        for key, value in self.__dict__.items():
            if key not in ["stack", "mask"]:
                all_attributes[key] = value
        with open(self.path + "Scan.pkl", "wb") as f:
            pickle.dump(all_attributes, f)

    def save_segmentation(self):
        save_mask(self.mask, self.path + "Segmentation")

    #%%
    # Segmentation methods
    def try_segmentation_settings(self, subset_size: int = 30):
        adjust_segmentation_parameters_on_subset(scan=self.stack,
                                                 subset_size=subset_size,
                                                 low_memory_mode=self.low_memory_mode)

    def set_segmentation_settings(self, settings: SegmentationSettings):
        self.segmentation_settings = settings

    def segment(self, logging: bool = True):
        if self.low_memory_mode:
            split_idxs = divide_scan(self.stack, size_gb=self.low_memory_mode)
            mask = np.zeros_like(self.stack, dtype=np.uint8)
            logging and print("Segmenting scan in low memory mode. \n"
                              "Splitting scan into {} parts of {:.2f} GB each."
                              .format(len(split_idxs) - 1,
                                      self.stack[split_idxs[0]:split_idxs[1]].nbytes / 1e9))
            for ii in tqdm(range(len(split_idxs) - 1)):
                left, right = split_idxs[ii], split_idxs[ii + 1]
                mask[left:right, :, :] = segment_scan(self.stack[left:right],
                                                      settings=self.segmentation_settings)
        else:
            logging and print("Segmenting scan in full memory mode.")
            mask = segment_scan(self.stack,
                                settings=self.segmentation_settings)

        self.mask = mask

    #%%
    # Analysis methods

    def calculate_properties(self):
        self.scan_dimensions_mm = self._calc_dimensions()
        self.V_mm3 = self._calc_volume()


    #%%
    # Utility methods
    def reslice(self,
                axis: tuple[int, int, int] | str = (1, 0, 2)):
        NotImplementedError("Reslicing not implemented yet.")
        if isinstance(axis, str):
            orientations: dict[str, int] = {"x": 0, "y": 1, "z": 2}
            np.where(self.slice_orientation == axis)
        self.stack = np.transpose(self.stack, axis)

    def show(self):
        viewer = napari.Viewer()
        viewer.add_image(self.stack, name="Scan")
        if self.mask is not None:
            viewer.add_labels(self.mask, name="Mask")

    def _calc_dimensions(self):
        h, w, d = self.stack.shape
        return (h * self.voxel_size_mm,
                w * self.voxel_size_mm,
                d * self.voxel_size_mm)

    def _calc_volume(self) -> float:
        return float(np.prod(self.scan_dimensions_mm))

    def _load_masks(self, logging: bool = False):
        segmentation_folder_exists = os.path.exists(self.path + "Segmentation")
        segmentation_folder_empty = len(os.listdir(self.path + "Segmentation")) == 0
        if not segmentation_folder_exists or segmentation_folder_empty:
            logging and print("No mask found at: {}".format(self.path + "Segmentation"))
            return

        self.mask = load_stack(self.path,
                               folder="Segmentation",
                               file_extension=".png",
                               logging=logging)

        logging and print("Loaded existing mask from: {}".format(self.path + "Segmentation"))

        if self.discard_ends:
            self.mask = self.mask[80:-80, :, :]
        if self.slice_range is not None:
            self.mask = self.mask[self.slice_range[0]:self.slice_range[1], :, :]

    def _load_stack(self, logging: bool = False):
        self.stack = load_stack(path=self.path,
                                folder="Slices",
                                logging=logging)
        if self.discard_ends:
            self.stack = self.stack[80:-80, :, :]
        if self.slice_range is not None:
            self.stack = self.stack[self.slice_range[0]:self.slice_range[1], :, :]

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
