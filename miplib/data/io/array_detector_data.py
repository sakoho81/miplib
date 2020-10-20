import os
import numpy as np
import pims
import itertools
from scipy.io import loadmat

from miplib.data.containers.array_detector_data import ArrayDetectorData
from miplib.data.containers.image import Image
from miplib.data.io import read as imread


def read_carma_mat(filename):
    """
    A simple implementation for the carma file import in Python
    :param filename: Path to the Carma .mat file
    :return: Returns a 2D nested list of miplib Image objects. The first dimension
             of the list corresponds to the Photosensors count and second, to the
             Detectors count
    """
    assert filename.endswith(".mat")
    #measurement = "meas_" + filename.split('/')[-1].split('.')[0]
    data = loadmat(filename)

    # Find measurement name (in case someone renamed the file)
    for key in list(data.keys()):
        if 'meas_' in key:
            data = data[key]
            break

    # Get necessary metadata
    spacing = list(data['PixelSize'][0][0][0][::-1])
    shape = list(data['Size'][0][0][0][::-1])

    detectors = int(data['DetectorsCount'][0][0][0])
    photosensors = len(data["PhotosensorsTime"][0][0][0])

    # Initialize data container
    container = ArrayDetectorData(detectors, photosensors)

    # Read images
    for i in range(photosensors):
        for j in range(detectors):
            name = 'pixel_d{}_p{}'.format(j, i)
            if shape[0] == 1:
                container[i, j] = Image(np.transpose(data[name][0][0])[0], spacing[1:])
            else:
                container[i, j] = Image(np.transpose(data[name][0][0]), spacing)


    return container

def read_airyscan_data(image_path, time_points=1, detectors=32):
    """ Read an Airyscan image. 
    
    Arguments:
        image_path {string} -- Path to the file
    
    Keyword Arguments:
        time_points {int} -- Number of time points (if a time series) (default: {1})
        detectors {int} -- Number of detectors (default: {32})
    
    Returns:
        ArrayDetectorData -- Returns the Airyscan image in the internal format for ISM
        processing.
    """
    # Open file
    data = pims.bioformats.BioformatsReader(image_path)
    
    # Get metadata
    spacing = [data.metadata.PixelsPhysicalSizeY(0), data.metadata.PixelsPhysicalSizeX(0)]
    
    # Initialize data container
    container = ArrayDetectorData(detectors, time_points)

    # Split time points
    data.bundle_axes = ['t', 'y', 'x']
    data = np.stack(np.split(data[0], time_points))

    # Save to data container
    for i in range(time_points):
        for j in range(detectors):
            container[i, j] = Image(data[i, j, :, :], spacing)
            
    return container


def read_tiff_sequence(path, detectors=25, channels=1):
    """
    Construct ArrayDetectorData from a series of TIF images on disk. The images
    should be named in a way that the detector channels are in a correct order
    ((det_0, channel_0), (det_0, channel_1),  (det_1, channel_0), (det_1, channel_1))
    after a simple sort.

    :param path: the directory that contains the images
    :param detectors: number of pixels in the array detectors
    :param channels: number of channels. Can denote photodetectors (pixel time split),
    color channels, time-points etc.

    :return: the ArrayDetectorData object that cotnains the imported data
    """

    files = sorted(filter(lambda x: x.endswith(".tif"), os.listdir(path)))
    if len(files) != detectors * channels:
        raise RuntimeError("The number of images does not match the data definition.")

    data = ArrayDetectorData(detectors, channels)
    steps = itertools.product(range(channels), range(detectors))
    for idx, (channel, detector) in enumerate(steps):
        image = imread.get_image(os.path.join(path, files[idx]), bioformats=False)
        data[detector, channel] = image

    return data
