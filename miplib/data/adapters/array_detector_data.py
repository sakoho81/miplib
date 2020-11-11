"""
Simple adapter wrappers that allow using ImageData and/or
Image objects in funcitons that were written for ArrayDetectorData.
"""

from miplib.data.containers.image import Image


class ImageDataAdapter(object):
    def __init__(self, data, kind="original", scale=100):
        self.data = data

        self.kind = kind
        self.scale = scale

        self.data.set_active_image(0, 0, self.scale, self.kind)

    @property
    def ndetectors(self):
        return self.data.series_count

    @property
    def ngates(self):
        return self.data.channel_count

    def __getitem__(self, item):
        gate, detector = item

        self.data.set_active_image(detector, gate, self.scale, self.kind)
        spacing = self.data.get_voxel_size()

        return Image(self.data[:], spacing)


class ImageAdapter(object):
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


class ArrayAdapter(object):
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



