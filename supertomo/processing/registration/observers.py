import SimpleITK as sitk
import matplotlib.pyplot as plt

from IPython.html.widgets import interact

from supertomo.utils import itkutils
from supertomo.ui.plots import image


def start_observer():
    """
    Start the registration observer. A list is initialized for the metric
    values
    """
    global metric_values

    metric_values = []


def update_observer(registration_method):
    """
    Adds the current metric value to the list.
    """
    global metric_values

    metric_values.append(registration_method.GetMetricValue())


def plot_result(self, fixed, moving):
    """
    Show registration results either in a Jupyter notebook, or Fiji.

    Parameters
    ----------
    fixed   The fixed image
    """
    global metric_values

    result = itkutils.resample_image(moving,
                                     self.transform,
                                     reference=fixed)

    plt.subplots(1, 2, figsize=(10, 8))
    fixed = itkutils.convert_to_numpy(sitk.Cast(fixed, sitk.sitkUInt8))[0]
    moving = itkutils.convert_to_numpy(
        sitk.Cast(result, sitk.sitkUInt8)
    )[0]

    def update(layer):
        # Plot metric values
        plt.subplot(1, 2, 1)
        plt.plot(metric_values, 'r')
        plt.title("Metric values")
        plt.xlabel('Iteration Number', fontsize=12)
        plt.ylabel('Metric Value', fontsize=12)

        # Plot image overlay
        plt.subplot(1, 2, 2)
        plt.title("Overlay")
        image.display_2d_image_overlay(fixed[layer, :, :], moving[layer, :, :])

    interact(update, layer=(0, fixed.shape[0] - 1, 1))

    del metric_values
