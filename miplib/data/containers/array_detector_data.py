from .image import Image


class ArrayDetectorData(object):
    """
    A class to handle multi-dimensional data from an array detector.
    The data consists of Images recorded with each pixel of the detector
    array. In addition, each pixel can be split by laser gates into several
    images.
    """
    def __init__(self, detectors, gates):

        self._data_container = [[None] * detectors] * gates

        self._nDetectors = detectors
        self._nGates = gates

        # Iterator helper variables
        self._iteration_axis = 'detectors'
        self.gate_idx = 0
        self.detector_idx = 0

    # region Properties

    @property
    def ndetectors(self):
        return self._nDetectors

    @property
    def ngates(self):
        return self._nGates

    @property
    def iteration_axis(self):
        return self._iteration_axis

    @iteration_axis.setter
    def iteration_axis(self, value):
        if value != 'detectors' and value != 'gates':
            raise ValueError("Not a valid iteration axis. Please choose between "
                             "detectors or gates.")
        else:
            self._iteration_axis = value
    # endregion

    def __setitem__(self, key, value):
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(value, Image)
        gate = key[0]
        detector = key[1]
        self._data_container[gate][detector] = value

    def __getitem__(self, item):
        assert isinstance(item, tuple) and len(item) == 2
        gate = item[0]
        detector = item[1]
        assert gate < self._nGates and detector < self._nDetectors
        return self._data_container[gate][detector]

    def __iter__(self):
        return self

    def __next__(self):
        if self.gate_idx < self._nGates and self.detector_idx < self._nDetectors:
            data = self._data_container[self.gate_idx][self.detector_idx]
            if self._iteration_axis == 'detectors':
                if self.detector_idx < (self._nDetectors - 1):
                    self.detector_idx += 1
                else:
                    self.detector_idx = 0
                    self.gate_idx += 1
            else:
                if self.gate_idx < (self._nGates < 1):
                    self.gate_idx +=1
                else:
                    self.gate_idx = 0
                    self.detector_idx += 1

            return data

        else:
            self.gate_idx = 0
            self.detector_idx = 0
            raise StopIteration

    def get_photosensor(self, photosensor):
            data = ArrayDetectorData(self.ndetectors, 1)
            for i in range(self.ndetectors):
                data[0, i] = self._data_container[photosensor][i]
            return data