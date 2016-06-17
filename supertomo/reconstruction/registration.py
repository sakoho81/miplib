"""
registration.py

Copyright (C) 2014 Sami Koho
All rights reserved.

This software may be modified and distributed under the terms
of the BSD license.  See the LICENSE file for details.

This file contains functions for registration of microscope image
volumes. The methods are based on Insight Toolkit (www.itk.org),
the installation of which is required in order to run the contained
functions

Currently (04/2014) rigid body registration of three-dimensional volumes
has been implemented. Several metrics 1. least-squares 2. viola-wells mutual
information 3. mattes mutual information are supported implemented

"""


import SimpleITK as sitk
import matplotlib.pyplot as plt
from IPython.display import clear_output

# PLOTS
# =============================================================================
# Plotting functions for showing the registration progress.


def start_plot():
    global metric_values

    metric_values = []


def end_plot():
    global metric_values

    del metric_values
    plt.close()


def plot_values(registration_method):
    global metric_values

    metric_values.append(registration_method.GetMetricValue())
    # clear the output area (wait=True, to reduce flickering), and plot current data
    clear_output(wait=True)
    # plot the similarity metric values
    plt.plot(metric_values, 'r')
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.show()

# REGISTRATION METHODS


# def itk_registration_rigid_3d(fixed_image, moving_image, options,
#                               initial_transform=None):
#     """
#     A Python implementation for a Rigid Body 3D registration, utilizing
#     ITK (www.itk.org) library functions.
#
#     :param fixed_image:
#                             The reference image. Must be an instance of
#                             itk.Image class. Numpy arrays can be converted
#                             to this format with ITK PyBuffer
#     :param moving_image:
#                             The image for which the spatial transform will
#                             be calculated. Same requirements as above.
#     :param options:
#                             Options provided by the user via CLI or the
#                             included GUI. See script_options.py.
#     :param initial_transform:
#                             Initial transform may be provided by the user
#                             instead of calculating it with an initializer.
#                             The transform must be of the type
#                             itk.VersorRigid3DTransform
#     :return:
#                             The final transform, an instance of
#                             itk.VersorRigid3DTransform
#     """
#
#     print 'Setting up registration job'
#
#     # Setup tempfile for saving registration progress
#     data_file_path = tempfile.mkdtemp('-supertomo.register')
#     data_to_save = ('Metric', 'X-versor', 'Y-versor', 'Z-versor', 'X-translation',
#                     'Y-translation', 'Z-translation')
#     data_file_name = os.path.join(data_file_path, 'registration_data.txt')
#     data_file = io.RowFile(data_file_name, data_to_save, append=False)
#     data_file.comment('Registration Command: %s' % (' '.join(map(str, sys.argv))))
#
#     # Check if type conversions are needed.
#     if options.use_internal_type is True:
#         image_type = options.internal_type+'3'
#     elif 'viola-wells' in options.registration_method:
#
#         print "Normalizing images for Viola-Wells"
#         # Cast to floating point. Can be single or double precision, depending
#         # on the setting
#         image_type = options.internal_type+'3'
#         fixed_image = itkutils.type_cast(
#             fixed_image,
#             options.image_type,
#             image_type
#         )
#         moving_image = itkutils.type_cast(
#             moving_image,
#             options.image_type,
#             image_type
#         )
#         # Normalize images
#         fixed_image = itkutils.normalize_image_filter(
#             fixed_image,
#             image_type
#         )
#         moving_image = itkutils.normalize_image_filter(
#             moving_image,
#             image_type
#         )
#     else:
#         image_type = options.image_type
#
#     fixed_image_region = fixed_image.GetBufferedRegion()
#
#     # REGISTRATION COMPONENTS SETUP
#     # ========================================================================
#     transform = itk.VersorRigid3DTransform.D.New()
#     optimizer = itk.VersorRigid3DTransformOptimizer.New()
#     interpolator = getattr(itk.LinearInterpolateImageFunction,
#                            image_type+'D').New()
#
#     registration = getattr(itk.ImageRegistrationMethod,
#                            image_type+image_type).New()
#
#     if 'mattes' in options.registration_method:
#         image_metric = getattr(itk.MattesMutualInformationImageToImageMetric,
#                                image_type+image_type).New()
#         image_metric.SetNumberOfHistogramBins(options.mattes_histogram_bins)
#         image_metric.SetNumberOfSpatialSamples(options.mattes_spatial_samples)
#     elif 'viola-wells' in options.registration_method:
#         image_metric = getattr(itk.MutualInformationImageToImageMetric,
#                                image_type+image_type).New()
#
#         image_metric.SetFixedImageStandardDeviation(options.vw_fixed_sd)
#         image_metric.SetMovingImageStandardDeviation(options.vw_moving_sd)
#
#         # Calculate number of spatial samples as a fraction of total pixel count
#         # of the fixed image
#         pixels = fixed_image_region.GetNumberOfPixels()
#         number_of_samples = int(options.vw_samples_multiplier*pixels)
#         image_metric.SetNumberOfSpatialSamples(number_of_samples)
#
#         optimizer.MaximizeOn()
#
#     elif 'mean-squared-difference' in options.registration_method:
#         image_metric = getattr(itk.MeanSquaresImageToImageMetric,
#                                image_type+image_type).New()
#     elif 'normalized-correlation' in options.registration_method:
#         image_metric = getattr(itk.NormalizedCorrelationImageToImageMetric,
#                                image_type+image_type).New()
#         image_metric.SetFixedImageRegion(fixed_image.GetBufferedRegion())
#     else:
#         print "Registration method %s not implemented" \
#               % options.registration_method
#         return None
#
#     # REGISTRATION
#     # ========================================================================
#     registration.SetMetric(image_metric)
#     registration.SetOptimizer(optimizer)
#     registration.SetTransform(transform)
#     registration.SetInterpolator(interpolator)
#
#     registration.SetFixedImage(fixed_image)
#     registration.SetMovingImage(moving_image)
#     registration.SetFixedImageRegion(fixed_image.GetBufferedRegion())
#
#     if initial_transform is None:
#         print 'Calculating initial registration parameters'
#         # INITIALIZER
#         # ====================================================================
#
#         initializer = getattr(
#             itk.CenteredVersorTransformInitializer,
#             image_type+image_type).New()
#
#         initializer.SetTransform(transform)
#         initializer.SetFixedImage(fixed_image)
#         initializer.SetMovingImage(moving_image)
#
#         if options.moments:
#             initializer.MomentsOn()
#
#         initializer.InitializeTransform()
#
#         # INITIAL PARAMETERS
#         # ====================================================================
#
#         # Initialize rotation
#         rotation = transform.GetVersor()
#         axis = rotation.GetAxis()
#
#         for i in range(len(axis)):
#             if i == options.set_rot_axis:
#                 axis[i] = 1.0
#             else:
#                 axis[i] = 0.0
#
#         rotation.Set(axis, double(options.set_rotation))
#
#         transform.SetRotation(rotation)
#
#     else:
#         transform = initial_transform
#
#     # Set initial parameters. The initializer will communicate directly with
#     # transform
#     registration.SetInitialTransformParameters(transform.GetParameters())
#
#     print 'Setting up optimizer'
#
#     # OPTIMIZER
#     # ========================================================================
#
#     # optimizer scale
#     translation_scale = 1.0 / options.translation_scale
#
#     optimizer_scales = itk.Array.D(transform.GetNumberOfParameters())
#     optimizer_scales.SetElement(0, 1.0)
#     optimizer_scales.SetElement(1, 1.0)
#     optimizer_scales.SetElement(2, 1.0)
#     optimizer_scales.SetElement(3, translation_scale)
#     optimizer_scales.SetElement(4, translation_scale)
#     optimizer_scales.SetElement(5, translation_scale)
#
#     optimizer.SetScales(double(optimizer_scales))
#
#     optimizer.SetMaximumStepLength(double(options.max_step_length))
#     optimizer.SetMinimumStepLength(double(options.min_step_length))
#     optimizer.SetNumberOfIterations(options.registration_max_iterations)
#
#     optimizer.SetRelaxationFactor(double(options.relaxation_factor))
#
#
#     # OBSERVER
#     # ========================================================================
#     def iteration_update():
#
#         current_parameter = transform.GetParameters()
#         optimizer_value = optimizer.GetValue()
#         if options.print_registration_progress:
#             print "M: %f   P: %f %f %f %f %f %f" % (
#                 optimizer_value,
#                 current_parameter.GetElement(0),
#                 current_parameter.GetElement(1),
#                 current_parameter.GetElement(2),
#                 current_parameter.GetElement(3),
#                 current_parameter.GetElement(4),
#                 current_parameter.GetElement(5)
#             )
#         tmplist = list()
#         tmplist.append(optimizer_value)
#         for j in range(len(current_parameter)):
#             tmplist.append(current_parameter.GetElement(j))
#         data_file.write(tmplist)
#
#     iteration_command = itk.PyCommand.New()
#     iteration_command.SetCommandCallable(iteration_update)
#     optimizer.AddObserver(itk.IterationEvent(), iteration_command)
#
#     # START
#     # ========================================================================
#
#     print "Starting registration"
#     registration.Update()
#
#     # Get the final parameters of the transformation
#     #
#     final_parameters = registration.GetLastTransformParameters()
#
#     versor_x = final_parameters[0]
#     versor_y = final_parameters[1]
#     versor_z = final_parameters[2]
#     final_translation_x = final_parameters[3]
#     final_translation_y = final_parameters[4]
#     final_translation_z = final_parameters[5]
#
#     number_of_iterations = optimizer.GetCurrentIteration()
#     best_value = optimizer.GetValue()
#
#     print "Final Registration Parameters "
#     print "Versor X  = %f" % versor_x
#     print "Versor Y = %f" % versor_y
#     print "Versor Z = %f" % versor_z
#     print "Translation X = %f" % final_translation_x
#     print "Translation Y = %f" % final_translation_y
#     print "Translation Z = %f" % final_translation_z
#     print "Iterations = %f" % number_of_iterations
#     print "Metric value = %f" % best_value
#
#     transform.SetParameters(final_parameters)
#     matrix = transform.GetMatrix()
#     offset = transform.GetOffset()
#
#     print "Matrix = "
#     for i in range(4):
#         print "%f %f %f %f" % (
#             matrix(i, 0),
#             matrix(i, 1),
#             matrix(i, 2),
#             matrix(i, 3)
#         )
#
#     print "Offset = "
#     print offset
#
#     # GET TRANSFORM
#     # ========================================================================
#
#     final_transform = itk.VersorRigid3DTransform.D.New()
#     final_transform.SetCenter(transform.GetCenter())
#     final_transform.SetParameters(final_parameters)
#     final_transform.SetFixedParameters(transform.GetFixedParameters())
#
#     data_file.close()
#
#     if options.show_plots:
#         plots.plot_rowfile(data_file_name, shape=330)
#
#     return final_transform


def itk_registration_2d(fixed_image, moving_image, options, initial_transform=None):
    """
    A Python implementation for a Rigid Body 2D registration, utilizing
    ITK (www.itk.org) library functions.

    :param fixed_image:
                            The reference image. Must be an instance of
                            itk.Image class. Numpy arrays can be converted
                            to this format with ITK PyBuffer
    :param moving_image:
                            The image for which the spatial transform will
                            be calculated. Same requirements as above.
    :param options:
                            Options provided by the user via CLI or the
                            included GUI. See script_options.py.
    :param tfm_type         Transform type string. Can be "rigid" or "similarity"
    :param initial_transform:
                            Initial transform may be provided by the user
                            instead of calculating it with an initializer.
                            The transform must be of the type
                            itk.CenteredRigid2DTransform
    :return:
                            The final transform, an instance of
                            itk.CenteredRigid2DTransform
    """
    print 'Setting up registration job'

    assert isinstance(fixed_image, sitk.Image)
    assert isinstance(moving_image, sitk.Image)

    # REGISTRATION COMPONENTS SETUP
    # ========================================================================

    registration = sitk.ImageRegistrationMethod()

    # OPTIMIZER
    registration.SetOptimizerAsRegularStepGradientDescent(
        options.max_step_length,
        options.min_step_length,
        options.registration_max_iterations,
        relaxationFactor=options.relaxation_factor
    )
    registration.SetOptimizerScalesFromPhysicalShift()

    # INTERPOLATOR
    registration.SetInterpolator(sitk.sitkLinear)

    # METRIC
    if options.registration_method == 'mattes':
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=options.mattes_histogram_bins
        )
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(options.mattes_sampling_percentage)

    elif options.registration_method == 'correlation':
        registration.SetMetricAsCorrelation()

    elif options.registration_method == 'mean-squared-difference':
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError("Unknown metric: %s" % options.registration_method)

    if initial_transform is None:
        print 'Calculating initial registration parameters'
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            sitk.Euler2DTransform() if options.tfm_type == "rigid" else sitk.Similarity2DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration.SetInitialTransform(transform)

    else:
        registration.SetInitialTransform(initial_transform)

    # OBSERVERS
    registration.AddCommand(sitk.sitkStartEvent, start_plot)
    registration.AddCommand(sitk.sitkEndEvent, end_plot)
    registration.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration))

    # START
    # ========================================================================

    print "Starting registration"
    final_transform = registration.Execute()

    print('Final metric value: {0}'.format(registration.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration.GetOptimizerStopConditionDescription()))

    return final_transform


# def itk_registration_similarity_2d(fixed_image, moving_image, options,
#                                 initial_transform=None):
#     """
#     A Python implementation for a Rigid Body 2D registration, utilizing
#     ITK (www.itk.org) library functions.
#
#     :param fixed_image:
#                             The reference image. Must be an instance of
#                             itk.Image class. Numpy arrays can be converted
#                             to this format with ITK PyBuffer
#     :param moving_image:
#                             The image for which the spatial transform will
#                             be calculated. Same requirements as above.
#     :param options:
#                             Options provided by the user via CLI or the
#                             included GUI. See script_options.py.
#     :param initial_transform:
#                             Initial transform may be provided by the user
#                             instead of calculating it with an initializer.
#                             The transform must be of the type
#                             itk.CenteredRigid2DTransform
#     :return:
#                             The final transform, an instance of
#                             itk.CenteredSimilarity2DTransform
#     """
#     print 'Setting up registration job'
#
#     # Setup tempfile for saving registration progress
#     data_file_path = tempfile.mkdtemp('-supertomo.register')
#     data_to_save = ('Metric', 'Rotation', 'CenterX', 'CenterY', 'X-translation',
#                     'Y-translation')
#     data_file_name = os.path.join(data_file_path, 'registration_data.txt')
#     data_file = io.RowFile(data_file_name, data_to_save, append=False)
#     data_file.comment('Registration Command: %s' % (' '.join(map(str, sys.argv))))
#
#     # Check if type conversions are needed.
#     if options.use_internal_type is True:
#         image_type = options.internal_type+'2'
#     elif 'viola-wells' in options.registration_method:
#
#         print "Normalizing images for Viola-Wells"
#         # Cast to floating point. Can be single or double precision, depending
#         # on the setting
#         image_type = options.internal_type+'2'
#         fixed_image = itkutils.type_cast(
#             fixed_image,
#             options.image_type,
#             image_type
#         )
#         moving_image = itkutils.type_cast(
#             moving_image,
#             options.image_type,
#             image_type
#         )
#         # Normalize images
#         fixed_image = itkutils.normalize_image_filter(
#             fixed_image,
#             image_type
#         )
#         moving_image = itkutils.normalize_image_filter(
#             moving_image,
#             image_type
#         )
#     else:
#         image_type = options.image_type
#
#     fixed_image_region = fixed_image.GetBufferedRegion()
#
#     # REGISTRATION COMPONENTS SETUP
#     # ========================================================================
#     transform = itk.CenteredSimilarity2DTransform.New()
#     optimizer = itk.RegularStepGradientDescentOptimizer.New()
#     interpolator = getattr(itk.LinearInterpolateImageFunction,
#                            image_type+'D').New()
#
#     registration = getattr(itk.ImageRegistrationMethod,
#                            image_type+image_type).New()
#
#     if 'mattes' in options.registration_method:
#         image_metric = getattr(itk.MattesMutualInformationImageToImageMetric,
#                                image_type+image_type).New()
#         image_metric.SetNumberOfHistogramBins(options.mattes_histogram_bins)
#         image_metric.SetNumberOfSpatialSamples(options.mattes_spatial_samples)
#     elif 'viola-wells' in options.registration_method:
#         image_metric = getattr(itk.MutualInformationImageToImageMetric,
#                                image_type+image_type).New()
#
#         image_metric.SetFixedImageStandardDeviation(options.vw_fixed_sd)
#         image_metric.SetMovingImageStandardDeviation(options.vw_moving_sd)
#
#         # Calculate number of spatial samples as a fraction of total pixel count
#         # of the fixed image
#         pixels = fixed_image_region.GetNumberOfPixels()
#         number_of_samples = int(options.vw_samples_multiplier*pixels)
#         print number_of_samples
#         image_metric.SetNumberOfSpatialSamples(number_of_samples)
#
#         optimizer.MaximizeOn()
#
#     elif 'normalized-correlation' in options.registration_method:
#         image_metric = getattr(itk.NormalizedCorrelationImageToImageMetric,
#                                image_type+image_type).New()
#         image_metric.SetFixedImageRegion(fixed_image.GetBufferedRegion())
#
#     elif 'mean-squared-difference' in options.registration_method:
#         image_metric = getattr(itk.MeanSquaresImageToImageMetric,
#                                image_type+image_type).New()
#     else:
#         print "Registration method %s not implemented" \
#               % options.registration_method
#         return None
#
#     # REGISTRATION
#     # ========================================================================
#     registration.SetMetric(image_metric)
#     registration.SetOptimizer(optimizer)
#     registration.SetTransform(transform)
#     registration.SetInterpolator(interpolator)
#
#     registration.SetFixedImage(fixed_image)
#     registration.SetMovingImage(moving_image)
#     registration.SetFixedImageRegion(fixed_image.GetBufferedRegion())
#
#     if initial_transform is None:
#         print 'Calculating initial registration parameters'
#         # INITIALIZER
#         # ====================================================================
#         init_type = 'CS2DTD'+image_type+image_type
#         initializer = getattr(itk.CenteredTransformInitializer, init_type).New()
#
#         initializer.SetTransform(transform)
#         initializer.SetFixedImage(fixed_image)
#         initializer.SetMovingImage(moving_image)
#
#         initializer.MomentsOn()
#
#         initializer.InitializeTransform()
#
#         transform.SetAngle(double(options.set_rotation))
#         transform.SetScale(double(options.set_scale))
#
#         # INITIAL PARAMETERS
#         # ====================================================================
#
#         registration.SetInitialTransformParameters(transform.GetParameters())
#
#     else:
#         assert isinstance(initial_transform, itk.CenteredRigid2DTransform)
#         registration.SetInitialTransformParameters(initial_transform.GetParameters())
#
#     # OPTIMIZER
#     # ========================================================================
#     print 'Setting up optimizer'
#     # optimizer scale
#     translation_scale = 1.0/options.translation_scale
#
#     optimizer_scales = itk.Array.D(transform.GetNumberOfParameters())
#     optimizer_scales.SetElement(0, options.scaling_scale)
#     optimizer_scales.SetElement(1, 1.0)
#     optimizer_scales.SetElement(2, translation_scale)
#     optimizer_scales.SetElement(3, translation_scale)
#     optimizer_scales.SetElement(4, translation_scale)
#     optimizer_scales.SetElement(5, translation_scale)
#
#
#     optimizer.SetScales(double(optimizer_scales))
#
#     optimizer.SetMaximumStepLength(double(options.max_step_length))
#     optimizer.SetMinimumStepLength(double(options.min_step_length))
#     optimizer.SetNumberOfIterations(options.registration_max_iterations)
#
#     optimizer.SetRelaxationFactor(double(options.relaxation_factor))
#
#
#     # OBSERVER
#     # ========================================================================
#
#     def iteration_update():
#
#         current_parameter = transform.GetParameters()
#         optimizer_value = optimizer.GetValue()
#         if options.print_registration_progress:
#             print "M: %f   P: %f %f %f %f %f %f" % (
#                 optimizer_value,
#                 current_parameter.GetElement(0),
#                 current_parameter.GetElement(1),
#                 current_parameter.GetElement(2),
#                 current_parameter.GetElement(3),
#                 current_parameter.GetElement(4),
#                 current_parameter.GetElement(5)
#             )
#         tmplist = list()
#         tmplist.append(optimizer_value)
#         for j in range(len(current_parameter)):
#             tmplist.append(current_parameter.GetElement(j))
#         data_file.write(tmplist)
#
#     iteration_command = itk.PyCommand.New()
#     iteration_command.SetCommandCallable(iteration_update)
#     optimizer.AddObserver(itk.IterationEvent(), iteration_command)
#
#     # START
#     # ========================================================================
#
#     print "Starting registration"
#     registration.Update()
#
#     # Get the final parameters of the transformation
#     #
#     final_parameters = registration.GetLastTransformParameters()
#
#     final_scale = final_parameters[0]
#     final_angle = final_parameters[1]
#     final_angle_in_degrees = final_angle*180.0/pi
#     final_rotation_center_x = final_parameters[2]
#     final_rotation_center_y = final_parameters[3]
#     final_translation_x = final_parameters[4]
#     final_translation_y = final_parameters[5]
#
#     number_of_iterations = optimizer.GetCurrentIteration()
#     best_value = optimizer.GetValue()
#
#     print "Final Registration Parameters "
#     print "Final scale = %f" % final_scale
#     print "Angle (degrees)  = %f" % final_angle_in_degrees
#     print "Center of rotation:"
#     print "X: %f" % final_rotation_center_x
#     print "Y: %f" % final_rotation_center_y
#     print "Translation X = %f" % final_translation_x
#     print "Translation Y = %f" % final_translation_y
#     print "Iterations = %f" % number_of_iterations
#     print "Metric value = %f" % best_value
#
#     transform.SetParameters(final_parameters)
#
#
#     # GET TRANSFORM
#     # ========================================================================
#
#     final_transform = itk.CenteredSimilarity2DTransform.New()
#     #final_transform.SetCenter(transform.GetCenter())
#     final_transform.SetParameters(final_parameters)
#     final_transform.SetFixedParameters(transform.GetFixedParameters())
#
#     data_file.close()
#
#     return final_transform
