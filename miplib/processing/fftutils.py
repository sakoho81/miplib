from typing import Any, Callable, Literal

import numpy as np

from miplib.data.containers.image import Image
from miplib.data.coordinates.polar import (
    PolarHighPassIndexer,
    PolarLowPassIndexer,
    SimplePolarIndexer,
)
from miplib.data.iterators.fourier_ring_iterators import FourierRingIterator
from miplib.data.iterators.fourier_shell_iterators import FourierShellIterator
from miplib.processing import ndarray, windowing

_WINDOW_FUNCS: dict[str, Callable[..., np.ndarray]] = {
    "tukey": windowing.apply_tukey_window,
    "hamming": windowing.apply_hamming_window,
}


def fft(
    array: np.ndarray,
    interpolation: float = 1.0,
    window: Literal["tukey", "hamming"] | None = "tukey",
    **kwargs: Any,
) -> np.ndarray:
    """Forward FFT with optional zero-padding interpolation and windowing."""
    if window is not None:
        func = _WINDOW_FUNCS.get(window)
        if func is None:
            raise ValueError(f"Unknown window type: {window!r}")
        array = func(array, **kwargs)

    if interpolation > 1.0:
        new_shape = tuple(int(interpolation * s) for s in array.shape)
        array = ndarray.expand_to_shape(array, new_shape)

    return np.fft.fftshift(np.fft.fftn(array))


def ifft(array_f: np.ndarray, interpolation: float = 1.0) -> np.ndarray:
    """Inverse FFT with optional interpolation for upsampling."""
    if interpolation > 1.0:
        new_shape = tuple(int(interpolation * s) for s in array_f.shape)
        array_f = ndarray.expand_to_shape(array_f, new_shape)

    return np.fft.ifftn(np.fft.ifftshift(array_f))


def ideal_fft_filter(image: Image, threshold: float, kind: str = "low") -> Image:
    """Ideal low-pass or high-pass frequency domain filter."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if not 0 < threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1.0")

    spacing = image.spacing
    fft_image = np.fft.fftshift(np.fft.fftn(image))

    if kind == "low":
        indexer: PolarLowPassIndexer | PolarHighPassIndexer = PolarLowPassIndexer(
            image.shape
        )
    elif kind == "high":
        indexer = PolarHighPassIndexer(image.shape)
    else:
        raise ValueError(f"Unknown filter kind: {kind!r}")

    r_max = int(np.floor(min(image.shape) / 2))
    fft_image *= indexer[threshold * r_max]

    return Image(np.abs(np.fft.ifftn(fft_image).real), spacing)


def butterworth_fft_filter(image: Image, threshold: float, n: int = 3) -> Image:
    """Low-pass Butterworth filter of order n."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if not 0 < threshold <= 1.0:
        raise ValueError("Cutoff frequency must be between 0 and 1.0")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be an integer >= 1")

    spacing = image.spacing
    r = SimplePolarIndexer(image.shape).r
    cutoff = threshold * image.shape[0]
    butter = 1.0 / (1.0 + (r / cutoff) ** (2 * n))

    fft_image = np.fft.fftshift(np.fft.fftn(image))
    fft_image *= butter

    return Image(np.abs(np.fft.ifftn(fft_image).real), spacing)


def gaussian_fft_filter(image: Image, threshold: float) -> Image:
    """Low-pass Gaussian frequency domain filter."""
    if not isinstance(image, Image):
        raise TypeError(f"Expected Image, got {type(image).__name__}")
    if not 0 < threshold <= 1.0:
        raise ValueError("Cutoff frequency must be between 0 and 1.0")

    spacing = image.spacing
    r = SimplePolarIndexer(image.shape).r
    r = r / image.shape[0]
    gauss = np.exp(-(r**2 / (2 * (threshold**2))))

    fft_image = np.fft.fftshift(np.fft.fftn(image))
    fft_image *= gauss

    return Image(np.abs(np.fft.ifftn(fft_image).real), spacing)


def power_spectrum(image: Image | np.ndarray) -> np.ndarray:
    """Centered N-dimensional power spectrum (|FFT|²)."""
    data = image[:] if isinstance(image, Image) else image
    return np.abs(fft(data, window=None)) ** 2


def frequency_axis(n: int, spacing: float) -> np.ndarray:
    """Physical frequency axis for FFT output: f = k / (n * spacing)."""
    return np.arange(n) / (n * spacing)


def radial_average(image: np.ndarray, bin_size: int = 2) -> np.ndarray:
    """Radial profile of 2D or 3D array.

    Dispatches to FourierRingIterator (2D) or FourierShellIterator (3D).
    """
    if image.ndim == 2:
        ring_iter = FourierRingIterator(image.shape, d_bin=bin_size)
        nbins = ring_iter.nbins
        averages = np.zeros(nbins)
        for ring_indices, ring_idx in ring_iter:
            subset = image[ring_indices]
            averages[ring_idx] = float(subset.sum()) / subset.size
        return averages
    elif image.ndim == 3:
        shell_iter = FourierShellIterator(image.shape, d_bin=bin_size)
        nbins = len(shell_iter.radii)
        averages = np.zeros(nbins)
        for shell_indices, shell_idx in shell_iter:
            subset = image[shell_indices]
            averages[shell_idx] = float(subset.sum()) / subset.size
        return averages
    else:
        raise ValueError(f"radial_average requires 2D or 3D array, got {image.ndim}D")


def power_spectrum_1d(
    image: Image | np.ndarray,
    bin_size: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """1D radial profile of power spectrum.

    Args:
        image: Input image (2D or 3D)
        bin_size: Bin size for radial averaging

    Returns:
        (frequencies, power) tuple where frequencies are in physical units if Image,
        otherwise normalized [0, 1]
    """
    data = image[:] if isinstance(image, Image) else image
    spacing = image.spacing[0] if isinstance(image, Image) else 1.0

    ps = power_spectrum(data)
    profile = radial_average(ps, bin_size)
    freq = frequency_axis(len(profile), spacing)
    return freq, profile
