import SimpleITK as sitk
import numpy

import miplib.processing.itk as ops_itk
from miplib.data.containers import image_data
from . import registration


# todo: This class has way too many responsibilities. Need to refactor at some point.
class RotatedMultiViewRegistration(object):
    """
    A class for multiview image registration. The method is based on
    functions inside the Insight Toolkit (www.itk.org), as in the original
    *miplib*. In *miplib* SimpleITK was used instead of Python
    wrapped ITK.

    The registration was updated to support multiple views
    and the new HDF5 data storage implementation. It was also implemented
    as a class.
    """

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

        # Fixed and moving image
        self.fixed_index = 0
        self.moving_index = 1

        # Results
        self.final_transform = None

        # REGISTRATION COMPONENTS SETUP
        # ========================================================================

        self.registration = sitk.ImageRegistrationMethod()

        # OPTIMIZER

        self.registration.SetOptimizerAsRegularStepGradientDescent(
            options.learning_rate,
            options.min_step_length,
            options.registration_max_iterations,
            relaxationFactor=options.relaxation_factor,
            estimateLearningRate=self.registration.EachIteration
        )
        # translation_scale = 1.0 / options.translation_scale
        # self.registration.SetOptimizerScales([10, 10, 10,
        #                                       .1, .1,.1])

        self.registration.SetOptimizerScalesFromJacobian()

        # INTERPOLATOR
        self.registration.SetInterpolator(sitk.sitkLinear)

        # METRIC
        if options.registration_method == 'mattes':
            self.registration.SetMetricAsMattesMutualInformation(
                numberOfHistogramBins=options.mattes_histogram_bins
            )

        elif options.registration_method == 'correlation':
            self.registration.SetMetricAsCorrelation()

        elif options.registration_method == 'mean-squared-difference':
            self.registration.SetMetricAsMeanSquares()
        else:
            raise ValueError("Unknown metric: %s" % options.registration_method)

        self.registration.SetMetricSamplingStrategy(self.registration.RANDOM)
        self.registration.SetMetricSamplingPercentage(
            options.sampling_percentage)

    def execute(self):
        """
        Run image registration. All the views are registered one by one. The
        image
        at index 0 is used as a reference.
        """

        # Get reference image.
        self.data.set_active_image(self.fixed_index,
                                   self.options.channel,
                                   self.options.scale,
                                   "original")
        fixed_image = self.data.get_itk_image()

        # Get moving image
        self.data.set_active_image(self.moving_index,
                                   self.options.channel,
                                   self.options.scale,
                                   "original")
        moving_image = self.data.get_itk_image()

        # INITIALIZATION
        # --------------
        # Start by rotating the moving image with the known rotation angle.
        print('Initializing registration')
        manual_transform = sitk.Euler3DTransform()

        # Rotate around the physical center of the image.
        rotation_center = moving_image.TransformContinuousIndexToPhysicalPoint(
            [(index - 1) / 2.0 for index in moving_image.GetSize()])
        manual_transform.SetCenter(rotation_center)

        # Rotation
        initial_rotation = self.data.get_rotation_angle(radians=True)
        if self.options.rot_axis == 0:
            manual_transform.SetRotation(initial_rotation, 0, 0)
        elif self.options.rot_axis == 1:
            manual_transform.SetRotation(0, initial_rotation, 0)
        else:
            manual_transform.SetRotation(0, 0, initial_rotation)

        # Translation
        manual_transform.SetTranslation([self.options.y_offset,
                                         self.options.x_offset,
                                         self.options.z_offset])

        modified_moving_image = ops_itk.resample_image(moving_image,
                                                       manual_transform)

        # 2. Run Automatic initialization

        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            modified_moving_image,
            sitk.AffineTransform(3),
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )

        # print "The initial transform is:"
        # print transform

        # Set initial transform
        self.registration.SetInitialTransform(transform)

        # SPATIAL MASK
        # =====================================================================
        # The registration metric works more reliably when it knows where
        # non-zero
        # voxels are located.
        thd = self.options.mask_threshold
        fixed_mask = sitk.BinaryDilate(
            sitk.BinaryThreshold(fixed_image, 0, thd, 0, 1))
        moving_mask = sitk.BinaryDilate(
            sitk.BinaryThreshold(modified_moving_image, 0, thd, 0, 1))

        self.registration.SetMetricFixedMask(fixed_mask)
        self.registration.SetMetricMovingMask(moving_mask)

        # START
        # ======================================================================

        if self.options.reg_enable_observers:
            # OBSERVERS
            self.registration.AddCommand(sitk.sitkStartEvent, registration.start_plot)
            self.registration.AddCommand(sitk.sitkIterationEvent, lambda: registration.plot_values(self.registration))

        print("Starting registration of views " \
              "%i (fixed) & %i (moving)" % (self.fixed_index, self.moving_index))

        result = self.registration.Execute(
            sitk.Cast(fixed_image, sitk.sitkFloat32),
            sitk.Cast(modified_moving_image, sitk.sitkFloat32))

        result = sitk.AffineTransform(result)
        # RESULTS
        # =====================================================================
        # Combine two partial transforms into one.
        # self.final_transform = sitk.Transform(manual_transform)
        # self.final_transform.AddTransform(result)

        # The two resulting transforms are combined into one here, because
        # it is easier to save a single transform into a HDF5 file.

        A0 = numpy.asarray(manual_transform.GetMatrix()).reshape(3, 3)
        c0 = numpy.asarray(manual_transform.GetCenter())
        t0 = numpy.asarray(manual_transform.GetTranslation())

        A1 = numpy.asarray(result.GetMatrix()).reshape(3, 3)
        c1 = numpy.asarray(result.GetCenter())
        t1 = numpy.asarray(result.GetTranslation())

        combined_mat = numpy.dot(A0, A1)
        combined_center = c1
        combined_translation = numpy.dot(A0, t1 + c1 - c0) + t0 + c0 - c1
        self.final_transform = sitk.AffineTransform(combined_mat.flatten(),
                                                    combined_translation,
                                                    combined_center)

        # Print final metric value and stopping condition
        print((
            'Final metric value: {0}'.format(self.registration.GetMetricValue())))
        print((
            'Optimizer\'s stopping condition, {0}'.format(
                self.registration.GetOptimizerStopConditionDescription())))
        print(self.final_transform)

        if self.options.reg_enable_observers:
            registration.end_plot(fixed_image, moving_image, self.final_transform)

    def set_moving_image(self, index):
        """
        Parameters
        ----------
        :param index    The moving image index from 0 to views number - 1

        """
        self.moving_index = index

    def set_fixed_image(self, index):
        """
        Parameters
        ----------
        :param index    The fixed image index from 0 to views number - 1. Should
                        be zero in most cases.

        """
        self.fixed_index = index

    def get_final_transform(self):
        """"
        Returns
        -------

        Get the final transform as an ITK transform
        """
        return self.final_transform

    def get_resampled_result(self):
        """

        Returns
        -------

        Get the registration result as a resampled image.
        """

        self.data.set_active_image(self.fixed_index,
                                   self.options.channel,
                                   self.options.scale,
                                   "original")

        fixed_image = self.data.get_itk_image()

        self.data.set_active_image(self.moving_index,
                                   self.options.channel,
                                   self.options.scale,
                                   "original")

        moving_image = self.data.get_itk_image()

        return ops_itk.resample_image(moving_image, self.final_transform,
                                      fixed_image)

    def save_result(self):
        scale = self.options.scale
        channel = self.options.channel
        view = self.moving_index

        # Add registered image
        self.data.set_active_image(view, channel, scale, "original")
        angle = self.data.get_rotation_angle(radians=False)
        spacing = self.data.get_voxel_size()
        registered_image = ops_itk.convert_from_itk_image(
            self.get_resampled_result())
        self.data.add_registered_image(registered_image, scale, view, channel,
                                       angle, spacing)
        # Add transform
        transform = self.get_final_transform()
        tfm_params = ops_itk.get_itk_transform_parameters(transform)
        self.data.add_transform(scale, view, channel, tfm_params[1],
                                tfm_params[2], tfm_params[0])

    def add_observers(self, start, update):
        """

        Parameters
        ----------
        start       Observer to add for the registration start
        update      Observer to add for registration progress updates.

        Returns
        -------

        """
        self.registration.AddCommand(sitk.sitkStartEvent, start)
        self.registration.AddCommand(sitk.sitkIterationEvent,
                                     lambda: update(self.registration))


class MultiViewRegistrationISM(RotatedMultiViewRegistration):

    def execute(self):
        # Get reference image.
        self.data.set_active_image(self.fixed_index,
                                   self.options.channel,
                                   self.options.scale,
                                   "original")
        fixed_image = self.data.get_itk_image()

        for idx in range(self.data.get_number_of_images("original")):
            print("Registering view {}".format(idx))
            self.moving_index = idx

            # Get moving image
            self.data.set_active_image(self.moving_index,
                                       self.options.channel,
                                       self.options.scale,
                                       "original")
            moving_image = self.data.get_itk_image()

            # Register
            self.final_transform = registration.itk_registration_rigid_2d(fixed_image,
                                                                          moving_image,
                                                                          self.options)
            self.save_result()

        print("All views registered and saved to the data structure")
