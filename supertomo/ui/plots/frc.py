import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.font_manager import FontProperties
from supertomo.data.containers.fourier_correlation_data import FourierCorrelationData, FourierCorrelationDataCollection


class FourierDataPlotter(object):
    def __init__(self, data):
        assert isinstance(data, FourierCorrelationDataCollection)

        self.data = data

        if len(self.data) < 3:
            self._columns = len(self.data)
        else:
            self._columns = 3

        if len(self.data) % self._columns == 0:
            self._rows = len(self.data) / self._columns
        else:
            self._rows = len(self.data) / self._columns + 1

    def plot_all(self):

        axescolor = '#f6f6f6'

        fig, plots = plt.subplots(self._rows, self._columns,
                                  facecolor=axescolor,
                                  figsize=(15, self._rows*4))
        # rect = fig.patch
        # rect.set_facecolor('white')

        fig.tight_layout()

        for dataset, plot in zip(self.data, plots):

            angle = dataset[0]
            data = dataset[1]

            title = "FRC @ angle %s" % angle

            self.__make_frc_subplot(plot, data, title)

        plt.show()

    def plot_one(self, angle):
        plt.figure(figsize=(5, 4))
        ax = plt.subplot(111)

        self.__make_frc_subplot(ax, self.data[int(angle)], "FRC at angle %s" % str(angle))

        plt.show()

    def __make_frc_subplot(self, ax, frc, title):
        """
        Creates a plot of the FRC curves in the curve_list. Single or multiple vurves can
        be plotted.
        """
        assert isinstance(frc, FourierCorrelationData)

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

        # Axis labelling
        xlabel = 'Spatial frequencies'
        ylabel = 'Fourier Ring Correlation (FRC)'
        ax.set_xlabel(xlabel, fontsize=12, position=(0.5, -0.2))
        ax.set_ylabel(ylabel, fontsize=12, position=(0.5, 0.5))
        ax.set_ylim([0, 1.2])

        # Title
        ax.set_title(title, fontsize=14)

        # Plot calculated FRC values as xy scatter.
        y = frc.correlation["correlation"]
        x = frc.correlation["frequency"]
        ax.plot(x, y, marker_array[0], markersize=6, color=colorArray[0],
                 label='FRC')

        # Plot polynomial fit as a line plot over the FRC scatter
        y = frc.correlation["curve-fit"]
        ax.plot(x, y, linewidth=3, color=colorArray[1],
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
        ax.plot(x, y, marker_array[3], color=colorArray[3], markersize=7,
                 label=label)

        # Plot resolution point
        y0 = frc.resolution["resolution-point"][0]
        x0 = frc.resolution["resolution-point"][1]

        ax.plot(x0, y0, 'ro', markersize=8, label='Resolution point')

        verts = [(x0, 0), (x0, y0)]
        xs, ys = zip(*verts)

        ax.plot(xs, ys, 'x--', lw=5, color='red', ms=10)
        #ax.text(x0, y0 + 0.10, 'RESOL-FREQ', fontsize=12)

        resolution = "The resolution is %f nm." % frc.resolution["resolution"]
        ax.text(0.5, -0.25, resolution, ha="center", fontsize=12)

        # Add legend
        ax.legend(loc='best')



