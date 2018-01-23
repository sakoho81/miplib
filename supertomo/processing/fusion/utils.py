import numpy

def get_coherent_images(psf, stack, dtype):
    """
    Return PSF and stack images so that they have
      - same orientation and voxel sizes
      - same fft-optimal shape
      - same floating point type
    and
      - the center of PSF is shifted to origin
      - PSF is normalized such that convolve (PSF, 1) = 1
    """
    psf_angle = psf.get_rotation_angle() or 0
    stack_angle = stack.get_rotation_angle() or 0

    psf_voxels = psf.get_voxel_sizes()
    stack_voxels = stack.get_voxel_sizes()

    psf_images = psf.images
    stack_images = stack.images

    if psf_angle != stack_angle:
        rotation = psf_angle - stack_angle
        psf_images = numpy.ndimage.rotate(psf_images, rotation, axes=(-1, -2))
        print 'PSF was rotated by', rotation

    if not numpy.allclose(psf_voxels, stack_voxels, rtol=0.01):
        zoom_factors = tuple([a / b for a, b in zip(psf_voxels, stack_voxels)])
        psf_images = numpy.ndimage.zoom(psf_images, zoom_factors)
        print 'PSF was zoomed by', zoom_factors

    max_shape = [max(a, b) for a, b in zip(psf_images.shape, stack_images.shape)]
    optimal_shape = tuple(map(FFTTasks.get_optimal_fft_size, max_shape))
    psf_images = expand_to_shape(psf_images, optimal_shape, dtype)
    stack_images = expand_to_shape(stack_images, optimal_shape, dtype)

    psf_images = fftpack.fftshift(psf_images)
    psf_images /= psf_images.sum()

    return psf_images, stack_images
