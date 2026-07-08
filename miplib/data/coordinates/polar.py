import numpy as np


def generate_polar_coordinate_grid(
    shape: tuple[int, ...], spacing: tuple[float, ...]
) -> tuple[np.ndarray, ...]:
    """Generate centered, physically scaled coordinate grid axes."""
    axes = tuple(
        np.arange(-np.floor(s / 2.0), np.ceil(s / 2.0)) * sp
        for s, sp in zip(shape, spacing)
    )
    return axes


class SimplePolarIndexer:
    """Basic indexer for a polar/spherical coordinate system.

    The ``r`` attribute contains radial distances from the array center,
    which can be indexed with ``[radius]`` to produce boolean masks.
    """

    def __init__(self, shape: tuple[int, ...]) -> None:
        if len(shape) not in (2, 3):
            raise ValueError(f"shape must be 2D or 3D, got {len(shape)}-dimensional")

        axes = tuple(np.arange(-np.floor(s / 2.0), np.ceil(s / 2.0)) for s in shape)
        self.meshgrid = np.meshgrid(*axes)
        self.r = np.sqrt(sum(axis**2 for axis in self.meshgrid))

    def __getitem__(self, item: float) -> np.ndarray:
        return self.r == item


class PolarLowPassIndexer(SimplePolarIndexer):
    """Low-pass mask: select points with radius less than the threshold."""

    def __getitem__(self, item: float) -> np.ndarray:
        return self.r < item


class PolarHighPassIndexer(SimplePolarIndexer):
    """High-pass mask: select points with radius greater than the threshold."""

    def __getitem__(self, item: float) -> np.ndarray:
        return self.r > item
