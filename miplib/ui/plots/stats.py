import numpy as np
import matplotlib.pyplot as plt


def plot_histogram(data, bins=50, figsize=(2,2)):
    """
    Calculate histogram for data

    :param data: some numpy.ndarray related datatype
    :param figsize: size of the plot (x,y)
    :param bins: the number of histogram bins
    :return: returns the Figure
    """
    assert issubclass(np.ndarray, data)

    fig, ax = plt.subplots(1,1, figsize=figsize)

    hist, bins = np.histogram(data.astype(np.uint16), bins=bins)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.bar(center, hist, align='center', width=width)

    return fig