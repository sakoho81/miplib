import matplotlib.pyplot as plt
plt.style.use("seaborn-colorblind")


def xy_scatter_plot_with_labels(x, y, labels, size=(3,3),
                                x_title=r"X-offset ($\mathrm{\mu m}$)",
                                y_title=r"Y-offset ($\mathrm{\mu m}$)"):

    assert len(x) == len(y) == len(labels)

    fig, ax = plt.subplots(figsize=size)
    ax.scatter(x, y)

    if x_title is not None:
        ax.set_xlabel(x_title)
    if y_title is not None:
        ax.set_ylabel(y_title)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    return fig
