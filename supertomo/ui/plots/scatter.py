import matplotlib.pyplot as plt


def xy_scatter_plot_with_labels(x, y, labels):

    assert len(x) == len(y) == len(labels)

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(labels):
        ax.annotate(txt, (x[i], y[i]))

    plt.show()