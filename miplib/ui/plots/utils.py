from matplotlib import pyplot as plt

def save_figure(figure, path, dpi=1200):
    """
    A really simple utility to save a figure to file.
    :param figure: a matplotlib.pyplot.Figure instance
    :param path:   a full path to the file. Make sure that the directory exists. The file type
                   will be decided according to the filename suffix.
    :param dpi:    dpi value for the plot
    :return:       nothing
    """
    assert isinstance(figure, plt.Figure)

    figure.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
