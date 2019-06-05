import numpy as np


def calculate_nearest_neighbor_distances(x_coords, y_coords):
    """
    Assumes an input of two arrays with coordinates of labels (e.g. centroids of blobs) and calculates
    the distance between each label pair.

    :param x_coords: the x coordinates
    :param y_coords: the y coordinates
    :return: Sorted list of distances (mean and std)
    """

    assert isinstance(x_coords, np.ndarray)
    assert isinstance(y_coords, np.ndarray)

    length = len(x_coords)

    x_s = np.broadcast_to(x_coords, (length,) * 2)
    y_s = np.broadcast_to(y_coords, (length,) * 2)

    distances = np.sqrt((x_s - np.transpose(x_s)) ** 2 + (y_s - np.transpose(y_s)) ** 2)

    distances = np.sort(distances, 0)

    mean_distances = np.mean(distances, axis=1)
    std_distances = np.std(distances, axis=1)

    return mean_distances, std_distances