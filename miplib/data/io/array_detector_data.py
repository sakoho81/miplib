import numpy as np
import pims
from scipy.io import loadmat

from miplib.data.containers.array_detector_data import ArrayDetectorData
from miplib.data.containers.image import Image


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

