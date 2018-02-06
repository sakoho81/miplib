import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from supertomo.data.containers.fourier_correlation import FourierCorrelationData


def plot_frc_curves(frc, title="FRC curve"):
    """
    Creates a plot of the FRC curves in the curve_list. Single or multiple vurves can
    be plotted.
    """
    assert isinstance(frc, FourierCorrelationData)

    # Define figure enviroment
    axescolor = '#f6f6f6'
    fig = plt.figure(1, facecolor=axescolor)
    rect = fig.patch
    rect.set_facecolor('white')

    ax = plt.subplot(111)

    # Font setting
    font0 = FontProperties()
    font1 = font0.copy()
    font1.set_size('medium')
    font = font1.copy()
    font.set_family('sans')
    rc('text', usetex=True)

    # Enable grid
    gridLineWidth = 0.2
    ax.yaxis.grid(True, linewidth=gridLineWidth, linestyle='-', color='0.05')

    # Marker setup
    colorArray = ['blue', 'green', 'red', 'orange', 'brown', 'black', 'violet', 'pink']
    marker_array = ['^', 's', 'o', 'd', '1', 'v', '*', 'p']

    # Axis limits
    plt.ylim([0, 1.2])

    # Axis labelling
    xlabel = 'Spatial frequencies'
    ylabel = 'Fourier Ring Correlation (FRC)'
    plt.xlabel(xlabel, fontsize=12, position=(0.5, -0.2))
    plt.ylabel(ylabel, fontsize=12, position=(0.5, 0.5))

    # Title
    plt.text(0.5, 1.06, title, horizontalalignment='center',
             fontsize=18, transform=ax.transAxes)

    # Plot calculated FRC values as xy scatter.
    y = frc.correlation["correlation"]
    x = frc.correlation["frequency"]
    plt.plot(x, y, marker_array[0], markersize=6, color=colorArray[0],
             label='FRC')

    # Plot polynomial fit as a line plot over the FRC scatter
    y = frc.correlation["curve-fit"]
    plt.plot(x, y, linewidth=3, color=colorArray[1],
             label='Least-squares fit')

    # Plot the resolution threshold curve
    y = frc.resolution["threshold"]
    res_crit = frc.resolution["criterion"]
    if res_crit == 'one-bit':
        label = 'One-bit curve'
    elif res_crit == 'half-bit':
        label = 'Half-bit curve'
    elif res_crit == 'fixed':
        label = 'y = %f' % y[0]
    else:
        raise AttributeError()
    plt.plot(x, y, marker_array[3], color=colorArray[3], markersize=7,
             label=label)

    # Plot resolution point
    y0 = frc.resolution["resolution-point"][0]
    x0 = frc.resolution["resolution-point"][1]

    plt.plot(x0, y0, 'ro', markersize=8, label='Resolution point')

    verts = [(x0, 0), (x0, y0)]
    xs, ys = zip(*verts)

    ax.plot(xs, ys, 'x--', lw=5, color='red', ms=10)
    ax.text(x0, y0 + 0.10, 'RESOL-FREQ', fontsize=12)

    # Add legend
    plt.legend(loc='best')

    plt.show()
