import numpy as np

from miplib.data.containers.image import Image
from miplib.processing.registration import registration
from scipy.ndimage import fourier_shift


def register_stack_slices(stack):
    """
    An utility to register slices in an image stack. The registration is performed
    by iterating over adjacent layers (0->1, 1->2,...). The shift obtained for each
    layer, is with respect to the image at idx=0.

    :param stack {Image}:  a 3D image stack
    :return: the image shifts with respect to index 0 (in pixels).

    """
    assert isinstance(stack, Image)
    assert stack.ndim == 3

    shifts = np.zeros((stack.shape[0], 2), dtype=np.float)

    for f_idx, m_idx in zip(range(0, stack.shape[0] - 1), range(1, stack.shape[0])):
        fixed = Image(stack[f_idx], stack.spacing[1:])
        moving = Image(stack[m_idx], stack.spacing[1:])

        offset = registration.phase_correlation_registration(fixed, moving, resample=False)
        shifts[m_idx] = shifts[f_idx] + np.asarray(offset)

    return shifts


def register_stack_slices_with_reference(stack, fixed):
    """
    Same as above, but the fixed image is provided by the user.

    :param stack {Image}:  a 3D image stack
    :param fixed: the image that is to be used as a
    reference for the registration
    :return: the image shifts with respect to the fixed (in pixels).
    """
    assert isinstance(stack, Image)
    assert stack.ndim == 3

    shifts = np.zeros((stack.shape[0], 2), dtype=np.float)

    for i in range(stack.shape[0]):
        moving = Image(stack[i], stack.spacing[1:])
        shifts[i] = registration.phase_correlation_registration(fixed, moving, resample=False)

    return shifts


def shift_stack_slices(stack, shifts):
    """
    Shift stack slices
    :param stack {Image}: a 3D stack
    :param shifts {np.ndarray}: a (stack_depth, 2) array with y,x shift defined on each
    row, corresponding to the relative of an image at postition i with respect to the first
    image in the stack (i=0)
    :returns {Image}: the resampled image with all the slices aligned.
    """

    assert isinstance(stack, Image)
    assert stack.ndim == 3

    if stack.shape[0] != shifts.shape[0]:
        raise ValueError("The shift array does not match the stack depth.")

    resampled = Image(np.zeros_like(stack), spacing=stack.spacing)

    for idx, (image, shift) in enumerate(zip(stack, shifts)):
        resampled[idx] = np.abs(np.fft.ifftn(fourier_shift(np.fft.fftn(image), shift)).real)

    return resampled