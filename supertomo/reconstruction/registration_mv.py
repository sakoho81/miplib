import SimpleITK as sitk
import matplotlib.pyplot as plt

from ..io import image_data
from ..ui import show
from ..utils import itkutils


class MultiViewRegistration:
    """
    A class for multiview image registration. The method is based on
    functions inside the Insight Toolkit (www.itk.org), as in the original
    *SuperTomo*. In *SuperTomo2* SimpleITK was used instead of Python
    wrapped ITK.

    The registration was updated to support multiple views
    and the new HDF5 data storage implementation. It was also implemented
    as a class.
    """

    @staticmethod
    def __get_user_input(message):
        """
        A method to ask question. The answer needs to be yes or no.

        Parameters
        ----------
        :param message  string, the question

        Returns
        -------

        Return a boolean: True for Yes, False for No
        """
        while True:
            answer = raw_input(message)
            if answer in ('y', 'Y', 'yes', 'YES'):
                return True
            elif answer in ('n', 'N', 'no', 'No'):
                return False
            else:
                print "Unkown command. Please state yes or no"

    def __init__(self, data, options):
        """
        :param data:    a ImageData object

        :param options: command line options that control the behavior
                            of the registration algorithm
         """
        assert isinstance(data, image_data.ImageData)

        # Parameters
        self.data = data
        self.options = options
        self.n_views = self.data.get_number_of_images("original")

        # Results
        self.result = None
        self.transform = None

        # REGISTRATION COMPONENTS SETUP
        # ========================================================================

        self.registration = sitk.ImageRegistrationMethod()

        # OPTIMIZER
        # TODO: The optimizer might need to be changed. This one is a bit stupid.
        self.registration.SetOptimizerAsRegularStepGradientDescent(
            options.max_step_length,
            options.min_step_length,
            options.registration_max_iterations,
            relaxationFactor=options.relaxation_factor
        )
        translation_scale = 1.0 / options.translation_scale

        self.registration.SetOptimizerScales([1.0, translation_scale, translation_scale])

        # INTERPOLATOR
        self.registration.SetInterpolator(sitk.sitkLinear)

        # METRIC
        if options.registration_method == 'mattes':
            self.registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=options.mattes_histogram_bins
            )
            self.registration.SetMetricSamplingStrategy(self.registration.RANDOM)
            self.registration.SetMetricSamplingPercentage(options.mattes_sampling_percentage)

        elif options.registration_method == 'correlation':
            self.registration.SetMetricAsCorrelation()

        elif options.registration_method == 'mean-squared-difference':
            self.registration.SetMetricAsMeanSquares()
        else:
            raise ValueError("Unknown metric: %s" % options.registration_method)

    def execute(self):
        """
        Run image registration. All the views are registered one by one. The image
        at index 0 is used as a reference.
        """
        # Get reference image.
        self.data.set_active_image(0, self.options.channel, self.options.scale,
                                   "original")
        fixed_image = self.data.get_itk_image()
        fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)

        # Iterate over the rotated views.
        for i in range(1, self.n_views):

            if self.data.check_if_exists("registered", i,
                                         self.options.channel, self.options.scale):
                if self.__get_user_input("A result already exists for the view %i. "
                                         "Do you want to skip registering it?"):
                    continue

            self.data.set_active_image(i, self.options.channel, self.options.scale,
                                       "original")
            moving_image = self.data.get_itk_image()
            moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

            initial_rotation = self.data.get_rotation_angle(radians=True)

            tx = sitk.Euler3DTransform()

            if self.options.rot_axis == 0:
                tx.SetAngle(initial_rotation, 0, 0)
            elif self.options.rot_axis == 1:
                tx.SetAngle(0, initial_rotation, 0)
            else:
                tx.SetAngle(0, 0, initial_rotation)

            if self.options.initializer:
                print 'Calculating initial registration parameters'
                transform = sitk.CenteredTransformInitializer(
                    fixed_image,
                    moving_image,
                    tx,
                    sitk.CenteredTransformInitializerFilter.MOMENTS
                )
                self.registration.SetInitialTransform(transform)
            else:
                tx.SetTranslation([self.options.y_offset,
                                   self.options.x_offset,
                                   self.options.z_offset])

                tx.SetCenter(itkutils.calculate_center_of_image(moving_image))
                self.registration.SetInitialTransform(tx)

            # OBSERVERS
            self.registration.AddCommand(sitk.sitkStartEvent, self.__start_plot)
            self.registration.AddCommand(sitk.sitkIterationEvent,
                                         lambda: self.__plot_values(self.registration))

            # START
            # ========================================================================

            print "Starting registration of view %i", i
            self.transform = self.registration.Execute(fixed_image, moving_image)

            # RESULTS
            self.result = itkutils.resample_image(moving_image,
                                                  self.transform,
                                                  reference=fixed_image),

            print('Final metric value: {0}'.format(self.registration.GetMetricValue()))
            print(
                'Optimizer\'s stopping condition, {0}'.format(self.registration.GetOptimizerStopConditionDescription()))

            # Show the result to the user or save directly. The latter makes sense
            # when one is relatively sure that the registration is going to work.
            if self.options.test_drive:
                self.__end_plot(fixed_image)
                if self.__get_user_input("Do you want to save the result (yes/no)? "):
                    self.save_result()
                else:
                    if self.__get_user_input("Do you want to continue with the registration?"):
                        print "Skipping view %i", self.data.get_active_image_index()
                        print "Continuing registration)"
                        continue
                    else:
                        print "Registration was aborted at view %i", self.data.get_active_image_index()
                        break
            else:
                self.save_result()

    def save_result(self):
        """
        Save registration result: resampled image and the spatial transformation
        """
        if isinstance(self.result, sitk.Image):
            self.result = itkutils.convert_to_numpy(
                sitk.Cast(self.result, sitk.sitkUInt8)
            )[0]

        self.data.add_registered_image(
            self.result,
            self.options.scale,
            self.data.get_active_image_index(),
            self.options.channel,
            self.data.get_rotation_angle(radians=False),
            self.data.get_voxel_size()
        )

        self.data.add_transform(self.options.scale,
                                self.data.get_active_image_index(),
                                self.options.channel,
                                self.transform.GetParameters(),
                                self.transform.GetFixedParameters(),
                                self.transform.GetName()
                                )

    def __start_plot(self):
        """
        Start the registration observer. A list is initialized for the metric
        values
        """
        self.metric_values = []

    def __end_plot(self, fixed):
        """
        Show registration results either in a Jupyter notebook, or Fiji.

        Parameters
        ----------
        fixed   The fixed image
        """
        if self.options.jupyter:
            plt.subplots(1, 2, figsize=(10, 8))
            fixed = itkutils.convert_to_numpy(sitk.Cast(fixed, sitk.sitkUInt8))[0]
            self.result = itkutils.convert_to_numpy(
                sitk.Cast(self.result, sitk.sitkUInt8)
            )[0]

            from IPython.html.widgets import interact

            def update(layer):
                # Plot metric values
                plt.subplot(1, 2, 1)
                plt.plot(self.metric_values, 'r')
                plt.title("Metric values")
                plt.xlabel('Iteration Number', fontsize=12)
                plt.ylabel('Metric Value', fontsize=12)

                # Plot image overlay
                plt.subplot(1, 2, 2)
                plt.title("Overlay")
                show.display_2d_image_overlay(fixed[layer, :, :], self.result[layer, :, :])

            interact(update, layer=(0, fixed.shape[0] - 1, 1))
        else:
            rgb_result = itkutils.make_composite_rgb_image(fixed, self.result)
            sitk.Show(rgb_result)

    def __plot_values(self, registration_method):
        """
        Adds the current metric value to the list.
        """
        self.metric_values.append(registration_method.GetMetricValue())

