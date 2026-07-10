"""Data source adapters for block-wise multi-view deconvolution."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from miplib.data.containers.image import Image
from miplib.data.containers.image_data import ImageData, ImageKey, ImageType
from miplib.processing.deconvolution.blocks import BlockSpec, extract_padded_block


class DataSource(Protocol):
    """Duck-typed contract for block-wise multi-view image access."""

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    n_views: int

    def get_image_block(self, view: int, block: BlockSpec) -> np.ndarray: ...
    def get_full_image(self, view: int) -> np.ndarray: ...


class ArrayDataSource:
    """Wraps a list of in-memory Images for testing and small datasets."""

    n_views: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]

    def __init__(self, images: list[Image]):
        if not images:
            raise ValueError("images must not be empty")
        self.n_views = len(images)
        self.shape = tuple(images[0].shape)
        self.spacing = tuple(images[0].spacing)
        self._images = [img.view(np.ndarray) for img in images]

    def get_image_block(self, view: int, block: BlockSpec) -> np.ndarray:
        return extract_padded_block(self._images[view], block)

    def get_full_image(self, view: int) -> np.ndarray:
        return self._images[view]


class ImageDataSource:
    """Lazy HDF5-backed data source for large multi-view datasets.

    Reads only the needed sub-blocks from disk on each call to
    ``get_image_block``, never loading full images into memory.
    """

    n_views: int
    shape: tuple[int, ...]
    spacing: tuple[float, ...]

    def __init__(
        self,
        data: ImageData,
        views: list[int],
        image_type: ImageType | str = ImageType.REGISTERED,
        channel: int = 0,
        scale: int = 100,
    ):
        if isinstance(image_type, str):
            image_type = ImageType(image_type)
        if not views:
            raise ValueError("views must not be empty")

        self._data = data
        self._views = list(views)
        self._image_type = image_type
        self._channel = channel
        self._scale = scale
        self.n_views = len(views)

        key0 = ImageKey(image_type, views[0], channel, scale)
        self.shape = data.get_image_size(key0)
        self.spacing = tuple(data.get_voxel_size(key0))

    def get_image_block(self, view: int, block: BlockSpec) -> np.ndarray:
        key = ImageKey(self._image_type, self._views[view], self._channel, self._scale)
        return self._data.get_registered_block(
            key, block.inner_size, block.pad, block.inner_start
        )

    def get_full_image(self, view: int) -> np.ndarray:
        key = ImageKey(self._image_type, self._views[view], self._channel, self._scale)
        return self._data.get_image_data(key)
