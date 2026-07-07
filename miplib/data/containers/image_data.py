import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import scipy.ndimage as ndimage

import miplib.data.image_data_operations as idops
import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image
from miplib.data.containers.image_data_store import (
    HDF5ImageStore,
    ImageDataStore,
    ImageKey,
    ImageType,
)

if TYPE_CHECKING:
    import SimpleITK as sitk

logger = logging.getLogger(__name__)


def _resolve_type(image_type: ImageType | str) -> ImageType:
    if isinstance(image_type, ImageType):
        return image_type
    return ImageType(image_type)


class ImageData:
    """Multi-view microscopy data backed by an HDF5 file.

    Compose an ``ImageDataStore`` for low-level I/O and expose convenience
    accessors for image registration, fusion, and analysis pipelines.
    """

    def __init__(self, path: str) -> None:
        self._store: ImageDataStore = HDF5ImageStore(path)

    # -- Read (convenience) ---------------------------------------------------

    def get_image(self, key: ImageKey) -> Image:
        data = self._store.get_image_data(key)
        attrs = self._store.get_image_attributes(key)
        return Image(data, attrs["spacing"])

    def get_image_data(self, key: ImageKey) -> np.ndarray:
        return self._store.get_image_data(key)

    def get_itk_image(self, key: ImageKey) -> "sitk.Image":
        data = self._store.get_image_data(key)
        attrs = self._store.get_image_attributes(key)
        return itkutils.convert_from_numpy(data, attrs["spacing"])

    def get_voxel_size(self, key: ImageKey) -> list[float]:
        return list(self._store.get_image_attributes(key)["spacing"])

    def get_image_size(self, key: ImageKey) -> tuple[int, ...]:
        size = self._store.get_image_attributes(key)["size"]
        return tuple(size)

    def get_max(self, key: ImageKey) -> float:
        return float(self._store.get_image_data(key).max())

    def get_dtype(self, key: ImageKey) -> np.dtype:
        return self._store.get_image_data(key).dtype

    def get_rotation_angle(self, key: ImageKey, radians: bool = True) -> float:
        angle = float(self._store.get_image_attributes(key)["angle"])
        return np.pi * angle / 180.0 if radians else angle

    def get_transform(self, key: ImageKey) -> "sitk.Transform":
        attrs = self._store.get_image_attributes(key)
        ndim = len(attrs["size"])
        return itkutils.make_itk_transform(
            attrs["tfm_type"], ndim, attrs["tfm_params"], attrs["tfm_fixed_params"]
        )

    def get_transform_parameters(
        self, key: ImageKey
    ) -> tuple[list[float], list[float], int]:
        attrs = self._store.get_image_attributes(key)
        return attrs["tfm_params"], attrs["tfm_fixed_params"], attrs["tfm_type"]

    def get_registered_block(
        self,
        key: ImageKey,
        block_size: np.ndarray,
        block_pad: int,
        block_start_index: np.ndarray,
    ) -> np.ndarray:
        return self._store.get_registered_block(
            key, block_size, block_pad, block_start_index
        )

    # -- Write ----------------------------------------------------------------

    def add_original_image(
        self,
        data: np.ndarray,
        scale: int,
        index: int,
        channel: int,
        angle: float,
        spacing: Sequence[float],
        chunk_size: tuple[int, ...] | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")

        if data.ndim == 3 and spacing[0] != spacing[1]:
            logger.info("Image index %d resampled for isotropic spacing.", index)
            z_zoom = spacing[0] / spacing[1]
            data = ndimage.zoom(data, (z_zoom, 1, 1), order=3)
            spacing = tuple(s if i != 0 else s / z_zoom for i, s in enumerate(spacing))

        self._store.add_image(
            ImageKey(ImageType.ORIGINAL, index, channel, scale),
            data,
            angle,
            spacing,
            chunk_size,
        )

    def add_registered_image(
        self,
        data: np.ndarray,
        scale: int,
        index: int,
        channel: int,
        angle: float,
        spacing: Sequence[float],
        chunk_size: tuple[int, ...] | None = None,
        overwrite: bool = False,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")

        key = ImageKey(ImageType.REGISTERED, index, channel, scale)
        if self._store.check_if_exists(key):
            if not overwrite:
                return
            self._store.delete_dataset(key)

        self._store.add_image(key, data, angle, spacing, chunk_size)

    def add_psf(
        self,
        data: np.ndarray,
        scale: int,
        index: int,
        channel: int,
        angle: float,
        spacing: Sequence[float],
        chunk_size: tuple[int, ...] | None = None,
        calculated: bool = False,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")

        self._store.add_image(
            ImageKey(ImageType.PSF, index, channel, scale),
            data,
            angle,
            spacing,
            chunk_size,
            calculated,
        )

    def add_fused_image(
        self,
        data: np.ndarray,
        channel: int,
        scale: int,
        spacing: Sequence[float],
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")

        self._store.add_fused_image(channel, scale, data, spacing)

    def add_transform(
        self,
        scale: int,
        index: int,
        channel: int,
        params: Sequence[float],
        fixed_params: Sequence[float],
        transform_type: int,
    ) -> None:
        self._store.add_transform(
            index, channel, scale, params, fixed_params, transform_type
        )

    # -- Query ----------------------------------------------------------------

    def get_number_of_images(self, image_type: ImageType | str) -> int:
        return self._store.get_number_of_images(_resolve_type(image_type))

    def get_scales(self, image_type: ImageType | str) -> list[int]:
        return self._store.get_scales(_resolve_type(image_type))

    def check_if_exists(
        self,
        image_type: ImageType | str,
        index: int,
        channel: int,
        scale: int,
    ) -> bool:
        return self._store.check_if_exists(
            ImageKey(_resolve_type(image_type), index, channel, scale)
        )

    # -- Derived image operations ---------------------------------------------

    def create_rescaled_images(
        self,
        image_type: ImageType | str,
        scale: int,
        chunk_size: tuple[int, ...] | None = None,
    ) -> None:
        idops.create_rescaled_images(
            self._store, _resolve_type(image_type), scale, chunk_size
        )

    def calculate_missing_psfs(self) -> None:
        idops.calculate_missing_psfs(self._store)

    def copy_registration_result(self, from_scale: int, to_scale: int) -> None:
        idops.copy_registration_result(self._store, from_scale, to_scale)

    # -- Lifecycle ------------------------------------------------------------

    @property
    def series_count(self) -> int:
        return self._store.series_count

    @property
    def channel_count(self) -> int:
        return self._store.channel_count

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> "ImageData":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
