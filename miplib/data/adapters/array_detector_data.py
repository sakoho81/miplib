"""
Simple adapter wrappers that allow using ImageData and/or
Image objects in funcitons that were written for ArrayDetectorData.
"""

from miplib.data.containers.image import Image
from miplib.data.containers.image_data import ImageKey, ImageType


class ImageDataAdapter:
    def __init__(self, data, kind=ImageType.ORIGINAL, scale=100):
        self.data = data
        self.kind = ImageType(kind) if isinstance(kind, str) else kind
        self.scale = scale

    @property
    def ndetectors(self):
        return self.data.series_count

    @property
    def ngates(self):
        return self.data.channel_count

    def __getitem__(self, item):
        gate, detector = item
        key = ImageKey(self.kind, detector, gate, self.scale)
        spacing = self.data.get_voxel_size(key)
        return Image(self.data.get_image_data(key), spacing)


class ImageAdapter:
    def __init__(self, data):
        self.data = data

    @property
    def ndetectors(self):
        return self.data.shape[1]

    @property
    def ngates(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        gate, detector = item

        return self.data[gate, detector]


class ArrayAdapter:
    def __init__(self, data, spacing):
        self.data = data
        self.spacing = spacing

    @property
    def ndetectors(self):
        return self.data.shape[1]

    @property
    def ngates(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        gate, detector = item

        return Image(self.data[gate, detector], self.spacing)
