from mt.ct_utils import *


class Scan:
    def __init__(self, path: str, low_memory_mode: float = 0.5):
        self.path: str = path

        self.scan: np.ndarray[np.uint16] | None = None
        self.mask: np.ndarray[np.uint8] | np.ndarray[np.uint32] | None = None

        self.voxel_size_mm: float | None = None
        self.scan_dimensions_mm: tuple[float, float, float] | None = None
        self.V: float | None = None

        self.slice_orientation: list = ["y", "z", "x"]

        self.segmentation_settings: SegmentationSettings = SegmentationSettings()
        self.low_memory_mode: float = low_memory_mode

    def __getitem__(self, item):
        return self.scan[item]

    def __str__(self):
        return (
                "Scan with shape {} and voxel size {:.2f}."
                .format(self.scan.shape, self.voxel_size_mm) +
                "\n Scanned volume: ({:.1f}x{:.1f}x{:.1f}) mm".format(*self.scan_dimensions_mm) +
                "\n Volume: {:.2f} mm^3".format(self.V)

        )

    def calculate_properties(self):
        self.scan_dimensions_mm = self._calc_dimensions()
        self.V = self._calc_volume()

    def reslice(self,
                axis: tuple[int, int, int] | str = (1, 0, 2)):
        NotImplementedError("Reslicing not implemented yet.")
        if isinstance(axis, str):
            orientations: dict[str, int] = {"x": 0, "y": 1, "z": 2}
            np.where(self.slice_orientation == axis)
        self.scan = np.transpose(self.scan, axis)

    def load(self,
             image_range: str | None = None,
             logging: bool = False):
        self.scan = load_stack(path=self.path,
                               image_range=image_range,
                               logging=logging)
        self.voxel_size_mm = read_scan_properties(self.path)

    def try_segmentation_settings(self, subset_size: int = 30):
        adjust_segmentation_parameters_on_subset(scan=self.scan,
                                                 subset_size=subset_size,
                                                 low_memory_mode=self.low_memory_mode)

    def set_segmentation_settings(self, settings: SegmentationSettings):
        self.segmentation_settings = settings

    def segment(self, logging: bool = True):
        if self.low_memory_mode:
            split_idxs = divide_scan(self.scan, size_gb=self.low_memory_mode)
            mask = np.zeros_like(self.scan, dtype=np.uint8)
            logging and print("Segmenting scan in low memory mode.")
            logging and print("Splitting scan into {} parts of {:.2f} GB each."
                              .format(len(split_idxs) - 1,
                                      self.scan[split_idxs[0]:split_idxs[1]].nbytes / 1e9))
            for ii in tqdm(range(len(split_idxs) - 1)):
                left, right = split_idxs[ii], split_idxs[ii + 1]
                mask[left:right, :, :] = segment_scan(self.scan[left:right],
                                                      settings=self.segmentation_settings)
        else:
            logging and print("Segmenting scan in full memory mode.")
            mask = segment_scan(self.scan,
                                settings=self.segmentation_settings)

        self.mask = mask

    def _calc_dimensions(self):
        h, w, d = self.scan.shape
        return (h * self.voxel_size_mm,
                w * self.voxel_size_mm,
                d * self.voxel_size_mm)

    def _calc_volume(self) -> float:
        return float(np.prod(self.scan_dimensions_mm))
