import numpy as np
import supertomo.processing.ops_array as ops_array

def find_points_interval(distance_map, start, stop):
    """
    Find the indexes of points located within a bin inside a distance map

    :param distance_map: Is a numpy.array consisting of distances of pixels
                         from the 0-frequency center in a centered 2D FFT
                         image
    :param start:        the ring/sphere starts at a radius r = start from the center
    :param stop:         the ring/sphere stops at a radious r = stop from the center
    :return:             returns a mask to select the indexes within the
                         specified interval.
    """
    arr_inf = distance_map >= start
    arr_sup = distance_map < stop
    ind = np.where(arr_inf * arr_sup)
    return ind

def find_directed_points_interval(distance_map, start, stop, angle, d_angle):

    # Calculate frequency bin (shell/ring)
    arr_inf = distance_map >= start
    arr_sup = distance_map < stop



class FourierShellIterator(object):

    def __init__(self, shape, **kwargs):
        """
        Generate a coordinate system for spherical indexing of 3D Fourier domain
        images

        :param shape: Shape of the image
        :param kwargs: Should contain at least the terms "d_theta" and
                      "d_bin", for angluar and radial bin sizes.

        """
        # Check that the two keywords are present. If not, the program will crash here.
        self.d_theta = kwargs["d_theta"]
        self.d_bin = kwargs["d_bin"]

        # Create Fourier grid
        axes = (np.arange(-np.floor(i / 2.0), np.ceil(i / 2.0)) for i in shape)
        z, x, y = np.meshgrid(*axes)

        # Create OP vector array
        self.r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        # Create inclination and azimuth angle arrays
        self.phi = np.arctan2(y, x)
        self.theta = np.arccos(ops_array.safe_divide(z, self.r))



    def __get_inclination_sector(self, theta_min, theta_max):

        arr_inf = self.theta >= theta_min
        arr_sup = self.theta < theta_max

        return arr_inf*arr_sup

    def __get_points_on_shell(self, shell_start, shell_stop):

        arr_inf = self.r >= shell_start
        arr_sup = self.r < shell_stop

        return arr_inf*arr_sup

    def get_points(self, index):  # type: (tuple) -> np.ndarray
        """
        Get coordinates of points in 3D space that lie on a section
        of the surface of a sphere, at a certain inclination angle.

        :param index: a tuple of two running indexes (bin, angle). The first one
                      is used to calculate the distance from the origin in
                      polar spherical coordinates, whereas the latter defines
                      the inclination angle.
        :return:      returns the indexes of points in the current section of
                      interest. The return value can be directly used to
                      index a 3D array (e.g. a centered Fourier image)
        """
        assert len(index) == 2, "You should give two indexes: bin / angle"

        bin = index[0]
        angle = index[1]
        shell = self.__get_points_on_shell(bin*self.d_bin, (bin+1)*self.d_bin)
        cone = self.__get_inclination_sector(angle*self.d_theta, (angle+1)*self.d_theta)

        return np.where(shell*cone)

