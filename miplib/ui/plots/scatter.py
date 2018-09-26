import matplotlib.pyplot as plt


def xy_scatter_plot_with_labels(x, y, labels, size=(3,3)):

    assert len(x) == len(y) == len(labels)

    fig, ax = plt.subplots(figsize=size)
    ax.scatter(x, y)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    return fig
