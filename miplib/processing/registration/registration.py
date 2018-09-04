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

import miplib.processing.itk as ops_itk
import miplib.ui.plots.image as show

import numpy as np

# region OBSERVERS

# PLOTS
# =============================================================================
# Plotting functions for showing the registration progress.


def start_plot():
    global metric_values

    metric_values = []


def end_plot(fixed, moving, transform):
    global metric_values
    plt.subplots(1, 2, figsize=(10, 8))

    # Plot metric values
    plt.subplot(1, 2, 1)
    plt.plot(metric_values, 'r')
    plt.title("Metric values")
    plt.xlabel('Iteration Number', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)

    # Plot image overlay
    resampled = ops_itk.resample_image(moving, transform, reference=fixed)

    if fixed.GetDimension() == 3:
        fixed = sitk.MaximumProjection(fixed, 2)[:, :, 0]
        resampled = sitk.MaximumProjection(resampled, 2)[:, :, 0]

    fixed = sitk.Cast(fixed, sitk.sitkUInt8)
    resampled = sitk.Cast(resampled, sitk.sitkUInt8)
    fixed = sitk.RescaleIntensity(fixed, 0, 255)
    resampled = sitk.RescaleIntensity(resampled, 0, 255)

    plt.subplot(1, 2, 2)
    plt.title("Overlay")
    show.display_2d_image_overlay(fixed, resampled)


    del metric_values


def plot_values(registration_method):
    global metric_values

    metric_values.append(registration_method.GetMetricValue())
    # clear the output area (wait=True, to reduce flickering), and plot current data
    # plot the similarity metric values


# endregion

# region RIGID SPATIAL DOMAIN REGISTRATION METHODS

#todo: Make a single method for n-dimensions. Too complicated now

def itk_registration_rigid_3d(fixed_image, moving_image, options):
    """
    A Python implementation for a Rigid Body 3D registration, utilizing
    ITK (www.itk.org) library functions.

    :param fixed_image:     The reference image. Must be an instance of
                            sitk.Image class.
    :param moving_image:    The image for which the spatial transform will
                            be calculated. Same requirements as above.
    :param options:         Options provided by the user via CLI or the
                            included GUI. See image_quality_options.py.
    :return:
                            The final transform as a sitk.Euler2DTransform
    """
    print 'Setting up registration job'

    assert isinstance(fixed_image, sitk.Image)
    assert isinstance(moving_image, sitk.Image)

    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # REGISTRATION COMPONENTS SETUP
    # ========================================================================

    registration = sitk.ImageRegistrationMethod()

    # OPTIMIZER
    registration.SetOptimizerAsRegularStepGradientDescent(
        options.learning_rate,
        options.min_step_length,
        options.registration_max_iterations,
        relaxationFactor=options.relaxation_factor,
        estimateLearningRate=registration.EachIteration
    )

    registration.SetOptimizerScalesFromJacobian()

    # translation_scale = 1.0/options.translation_scale
    # registration.SetOptimizerScales([1.0, translation_scale, translation_scale])

    # INTERPOLATOR
    registration.SetInterpolator(sitk.sitkLinear)

    # METRIC
    if options.registration_method == 'mattes':
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=options.mattes_histogram_bins
        )
    elif options.registration_method == 'correlation':
        registration.SetMetricAsCorrelation()

    elif options.registration_method == 'mean-squared-difference':
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError("Unknown metric: %s" % options.registration_method)

    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(options.sampling_percentage)

    if options.reg_translate_only:
        tx = sitk.TranslationTransform(3)
    else:

        tx = sitk.Euler3DTransform()

        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            tx,
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        registration.SetInitialTransform(transform)

        tx.SetCenter(ops_itk.calculate_center_of_image(moving_image))

    registration.SetInitialTransform(tx)

    if options.reg_enable_observers:
        # OBSERVERS
        registration.AddCommand(sitk.sitkStartEvent, start_plot)
        registration.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration))

    # START
    # ========================================================================

    print "Starting registration"
    final_transform = registration.Execute(fixed_image, moving_image)

    print('Final metric value: {0}'.format(registration.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration.GetOptimizerStopConditionDescription()))

    if options.reg_enable_observers:
        end_plot(fixed_image, moving_image, final_transform)

    return final_transform


def itk_registration_rigid_2d(fixed_image, moving_image, options):
    """
    A Python implementation for a Rigid Body 2D registration, utilizing
    ITK (www.itk.org) library functions.

    :param fixed_image:     The reference image. Must be an instance of
                            sitk.Image class.
    :param moving_image:    The image for which the spatial transform will
                            be calculated. Same requirements as above.
    :param options:         Options provided by the user via CLI or the
                            included GUI. See image_quality_options.py.
    :return:
                            The final transform as a sitk.Euler2DTransform
    """
    print 'Setting up registration job'

    assert isinstance(fixed_image, sitk.Image)
    assert isinstance(moving_image, sitk.Image)

    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

    # REGISTRATION COMPONENTS SETUP
    # ========================================================================

    registration = sitk.ImageRegistrationMethod()

    # OPTIMIZER
    registration.SetOptimizerAsRegularStepGradientDescent(
        options.learning_rate,
        options.min_step_length,
        options.registration_max_iterations,
        relaxationFactor=options.relaxation_factor,
        estimateLearningRate=registration.EachIteration
    )

    translation_scale = 1.0/options.translation_scale
    registration.SetOptimizerScales([1.0, translation_scale, translation_scale])

    # INTERPOLATOR
    registration.SetInterpolator(sitk.sitkLinear)

    # METRIC
    if options.registration_method == 'mattes':
        registration.SetMetricAsMattesMutualInformation(
            numberOfHistogramBins=options.mattes_histogram_bins
        )
    elif options.registration_method == 'correlation':
        registration.SetMetricAsCorrelation()

    elif options.registration_method == 'mean-squared-difference':
        registration.SetMetricAsMeanSquares()
    else:
        raise ValueError("Unknown metric: %s" % options.registration_method)

    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(options.sampling_percentage)

    if options.reg_translate_only:
        tx = sitk.TranslationTransform(2)
    else:

        tx = sitk.Euler2DTransform()
        tx.SetAngle(options.set_rotation)
        if options.initializer:
            print 'Calculating initial registration parameters'
            transform = sitk.CenteredTransformInitializer(
                fixed_image,
                moving_image,
                tx,
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration.SetInitialTransform(transform)

        else:
            tx.SetTranslation([options.y_offset, options.x_offset])

            tx.SetCenter(ops_itk.calculate_center_of_image(moving_image))
    registration.SetInitialTransform(tx)

    if options.reg_enable_observers:
        # OBSERVERS
        registration.AddCommand(sitk.sitkStartEvent, start_plot)
        registration.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration))

    # START
    # ========================================================================

    print "Starting registration"
    final_transform = registration.Execute(fixed_image, moving_image)

    print('Final metric value: {0}'.format(registration.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration.GetOptimizerStopConditionDescription()))

    if options.reg_enable_observers:
        end_plot(fixed_image, moving_image, final_transform)

    return final_transform

# endregion

# region DEFORMABLE SPATILA DOMAIN REGISTRATION METHDOS

def itk_registration_similarity_2d(fixed_image, moving_image, options):
    """
    A Python implementation for a Rigid Body 2D registration, utilizing
    ITK (www.itk.org) library functions.

    :param fixed_image:     The reference image. Must be an instance of
                            sitk.Image class.
    :param moving_image:    The image that is to be registered. Must be
                            an instance of sitk.Image class.
                            The image for which the spatial transform will
                            be calculated. Same requirements as above.
    :param options:         Options provided by the user via CLI

    :return:                The final transform as a sitk.Similarity2DTransform
    """
    print 'Setting up registration job'

    assert isinstance(fixed_image, sitk.Image)
    assert isinstance(moving_image, sitk.Image)

    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

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
    translation_scale = 1.0 / options.translation_scale
    scaling_scale = 1.0 / options.scaling_scale

    registration.SetOptimizerScales([scaling_scale, 1.0, translation_scale, translation_scale])

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

    print 'Calculating initial registration parameters'
    tx = sitk.Similarity2DTransform()
    tx.SetAngle(options.set_rotation)
    tx.SetScale(options.set_scale)

    if options.initializer:
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            tx,
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )
        registration.SetInitialTransform(transform)
    else:
        tx.SetTranslation([options.y_offset, options.x_offset])
        tx.SetCenter(ops_itk.calculate_center_of_image(moving_image))
        registration.SetInitialTransform(tx)

    # OBSERVERS

    registration.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration.AddCommand(sitk.sitkEndEvent, end_plot)
    registration.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration))

    # START
    # ========================================================================

    print "Starting registration"
    final_transform = registration.Execute(fixed_image, moving_image)

    print('Final metric value: {0}'.format(registration.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration.GetOptimizerStopConditionDescription()))

    end_plot(fixed_image, moving_image, final_transform)

    return final_transform


def itk_registration_affine_2d(fixed_image, moving_image, options):
    """
    A Python implementation for a Rigid Body 2D registration, utilizing
    ITK (www.itk.org) library functions.

    :param fixed_image:     The reference image. Must be an instance of
                            sitk.Image class.
    :param moving_image:    The image that is to be registered. Must be
                            an instance of sitk.Image class.
                            The image for which the spatial transform will
                            be calculated. Same requirements as above.
    :param options:         Options provided by the user via CLI

    :return:                The final transform as a sitk.Similarity2DTransform
    """
    print 'Setting up registration job'

    assert isinstance(fixed_image, sitk.Image)
    assert isinstance(moving_image, sitk.Image)

    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)

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
    translation_scale = 1.0 / options.translation_scale
    scaling_scale = 1.0 / options.scaling_scale

    registration.SetOptimizerScales([scaling_scale, 1.0, translation_scale, translation_scale])

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

    print 'Calculating initial registration parameters'
    tx = sitk.AffineTransform()


    if options.initializer:
        transform = sitk.CenteredTransformInitializer(
            fixed_image,
            moving_image,
            tx,
            sitk.CenteredTransformInitializerFilter.MOMENTS
        )
        registration.SetInitialTransform(transform)
    else:
        tx.SetTranslation([options.y_offset, options.x_offset])
        tx.SetCenter(ops_itk.calculate_center_of_image(moving_image))
        registration.SetInitialTransform(tx)

    # OBSERVERS

    registration.AddCommand(sitk.sitkStartEvent, start_plot)
    #registration.AddCommand(sitk.sitkEndEvent, end_plot)
    registration.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration))

    # START
    # ========================================================================

    print "Starting registration"
    final_transform = registration.Execute(fixed_image, moving_image)

    print('Final metric value: {0}'.format(registration.GetMetricValue()))
    print('Optimizer\'s stopping condition, {0}'.format(registration.GetOptimizerStopConditionDescription()))

    end_plot(fixed_image, moving_image, final_transform)

    return final_transform

# endregion

# region RIGID FREQUENCY DOMAIN REGISTRATION METHODS


def phase_correlation_registration(fixed_image, moving_image):
    """
    A simple Phase Correlation based image registration method.
    :param fixed_image: the reference image as sitk.Image object
    :param moving_image: the moving image as sitk.Image object
    :return: returns the SimpleITK transform
    """

    assert isinstance(fixed_image, sitk.Image)
    assert isinstance(moving_image, sitk.Image)

    spacing = fixed_image.GetSpacing()

    # Make a mask, not used here. May help in making this more robust
    mask = ops_itk.convert_from_numpy(np.full(fixed_image.GetSize()[::-1], 255),
                                      spacing)

    # Calculate cross correlation
    correlation = sitk.MaskedFFTNormalizedCorrelation(fixed_image, mask,
                                                      moving_image, mask)
    # Find index of the maximum
    array = ops_itk.convert_from_itk_image(correlation)
    image_size = fixed_image.GetSize()[::-1]
    offset = np.unravel_index(np.argmax(array), image_size)

    # Convert offset to physical spacing. Also consider that it can be in positive
    # and negative direction
    offset = tuple(size - offset if offset > size / 2 else -offset
                   for size, offset in zip(image_size, offset))

    offset = tuple(shift * step for shift, step in zip(offset, spacing))

    # Make transform
    transform = sitk.TranslationTransform(2)
    transform.SetOffset(offset)

    return transform

# endregion
