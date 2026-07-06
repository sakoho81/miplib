from .image import Image


class _ArrayDetectorDataIterator:
    """Iterator for ArrayDetectorData, supporting multiple independent traversals."""

    def __init__(self, data: "ArrayDetectorData"):
        self._data = data
        self._gate_idx = 0
        self._detector_idx = 0

    def __next__(self):
        if (
            self._gate_idx < self._data._n_gates
            and self._detector_idx < self._data._n_detectors
        ):
            value = self._data._data_container[self._gate_idx][self._detector_idx]
            if self._data._iteration_axis == "detectors":
                if self._detector_idx < self._data._n_detectors - 1:
                    self._detector_idx += 1
                else:
                    self._detector_idx = 0
                    self._gate_idx += 1
            else:
                if self._gate_idx < self._data._n_gates - 1:
                    self._gate_idx += 1
                else:
                    self._gate_idx = 0
                    self._detector_idx += 1
            return value
        raise StopIteration


class ArrayDetectorData:
    """Container for multi-dimensional data from an array detector.

    Stores Images recorded with each pixel of the detector array. Each pixel
    can be split by laser gates into multiple images. Data is indexed as
    ``[gate, detector]`` where *gate* is the first index and *detector* is
    the second. Note: the constructor takes ``(detectors, gates)`` in the
    opposite order to the indexing convention.
    """

    def __init__(self, detectors: int, gates: int) -> None:
        self._data_container = [[None] * detectors for _ in range(gates)]
        self._n_detectors = detectors
        self._n_gates = gates
        self._iteration_axis = "detectors"

    @property
    def ndetectors(self) -> int:
        return self._n_detectors

    @property
    def ngates(self) -> int:
        return self._n_gates

    @property
    def iteration_axis(self) -> str:
        return self._iteration_axis

    @iteration_axis.setter
    def iteration_axis(self, value: str) -> None:
        if value not in ("detectors", "gates"):
            raise ValueError(
                "Not a valid iteration axis. Please choose between detectors or gates."
            )
        self._iteration_axis = value

    def __setitem__(self, key: tuple[int, int], value: Image) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Index must be a 2-tuple (gate, detector)")
        if not isinstance(value, Image):
            raise TypeError(f"Value must be an Image, got {type(value).__name__}")
        gate, detector = key
        self._data_container[gate][detector] = value

    def __getitem__(self, key: tuple[int, int]) -> Image:
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError("Index must be a 2-tuple (gate, detector)")
        gate, detector = key
        if gate >= self._n_gates or detector >= self._n_detectors:
            raise IndexError(
                f"Index ({gate}, {detector}) out of range for "
                f"({self._n_gates} gates, {self._n_detectors} detectors)"
            )
        return self._data_container[gate][detector]

    def __iter__(self):
        return _ArrayDetectorDataIterator(self)

    def get_photosensor(self, photosensor: int) -> "ArrayDetectorData":
        data = ArrayDetectorData(self.ndetectors, 1)
        for i in range(self.ndetectors):
            data[0, i] = self._data_container[photosensor][i]
        return data
