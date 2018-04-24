import itertools
import numpy as np
from supertomo.data.containers.array_detector_data import ArrayDetectorData


def make_template_image(data, imagesz=250):
    assert isinstance(data, ArrayDetectorData)

    blocksz = imagesz/int(np.sqrt(data.ndetectors))

    # First calculate the total photon count for images from each detector
    # and photosensor
    pixels = np.zeros(data.ndetectors*data.ngates)

    data.iteration_axis = 'detectors'
    for idx, image in enumerate(data):
        pixels[idx] = image.sum()

    # Then generate a template image for each photosensor
    container = []
    for gate in range(data.ngates):
        image = np.zeros((imagesz, imagesz))
        idx = 0
        for x, y in itertools.product(xrange(0, imagesz, blocksz), xrange(0, imagesz, blocksz)):
            pixel_index = gate*data.ndetectors + idx
            image[x:x + blocksz, y:y + blocksz] = pixels[pixel_index]
            idx += 1

        container.append(image)

    return container






