import os
from multiprocessing import Queue

from miplib.data.containers.image import Image
from miplib.data.io import write as imwrite


class ImageWriterBase:
    def write(self, image):
        pass


class QueuedImageWriter(ImageWriterBase):
    def __init__(self, queue):
        assert isinstance(queue, Queue)

        self.queue = queue

    def write(self, image):
        assert isinstance(image, Image)

        self.queue.put(image)


class TiffImageWriter(ImageWriterBase):
    def __init__(self, directory):
        self.index = 0
        self.dir = directory

    def __get_full_path(self):
        filename = f"result_{self.index}.tif"
        return os.path.join(self.dir, filename)

    def write(self, image):
        assert isinstance(image, Image)

        imwrite.image(self.__get_full_path(), image)

        self.index += 1
