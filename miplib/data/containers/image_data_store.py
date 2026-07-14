import logging
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np

import miplib.processing.ndarray as arrayutils

logger = logging.getLogger(__name__)


class ImageType(StrEnum):
    """Kind of image stored in the dataset."""

    ORIGINAL = "original"
    REGISTERED = "registered"
    FUSED = "fused"
    PSF = "psf"


@dataclass(frozen=True)
class ImageKey:
    """Unique identifier for an image dataset.

    Parameters
    ----------
    image_type : The kind of image (original, registered, fused, psf).
    index : View number (0-based).
    channel : Color channel (0-based), default 0.
    scale : Size as percentage of full resolution, default 100.
    """

    image_type: ImageType
    index: int
    channel: int = 0
    scale: int = 100

    def to_path(self) -> str:
        """Return the HDF5 path for this key."""
        if self.image_type == ImageType.FUSED:
            return f"fused/channel_{self.channel}_scale_{self.scale}"
        return (
            f"{self.image_type.value}/{self.index}"
            f"/channel_{self.channel}_scale_{self.scale}"
        )


class ImageDataStore(Protocol):
    """Storage interface for multi-view image data.

    Implementations must handle create/read/update/delete of image datasets,
    attributes (spacing, angle, transforms), and scaled/blocks access.
    """

    @property
    def series_count(self) -> int:
        """Number of distinct original image views (indices)."""
        ...

    @property
    def channel_count(self) -> int:
        """Number of color channels in the dataset."""
        ...

    def add_image(
        self,
        key: ImageKey,
        data: np.ndarray,
        angle: float,
        spacing: Sequence[float],
        chunk_size: tuple[int, ...] | None = None,
        calculated: bool | None = None,
    ) -> None:
        """Store an image array with metadata.

        No-op if the dataset already exists at the given key.
        """
        ...

    def add_fused_image(
        self,
        channel: int,
        scale: int,
        data: np.ndarray,
        spacing: Sequence[float],
    ) -> None:
        """Store a fused (reconstructed) image.

        Fused images are stored outside the per-view hierarchy.
        """
        ...

    def add_transform(
        self,
        index: int,
        channel: int,
        scale: int,
        params: Sequence[float],
        fixed_params: Sequence[float],
        transform_type: int,
    ) -> None:
        """Attach spatial transform parameters to a registered view."""
        ...

    def get_image_data(self, key: ImageKey) -> np.ndarray:
        """Read the raw image array at *key*."""
        ...

    def get_image_attributes(self, key: ImageKey) -> dict[str, Any]:
        """Read all metadata attributes (spacing, angle, size, transforms)."""
        ...

    def get_registered_block(
        self,
        key: ImageKey,
        block_size: np.ndarray,
        block_pad: int,
        block_start_index: np.ndarray,
    ) -> np.ndarray:
        """Read a padded sub-block of a registered image.

        Handles boundary conditions by zero-padding regions outside the
        image extent.
        """
        ...

    def get_number_of_images(self, image_type: ImageType) -> int:
        """Number of view indices present for *image_type*."""
        ...

    def get_scales(self, image_type: ImageType) -> list[int]:
        """Available scale levels for *image_type*.

        Raises ``ValueError`` when scales are inconsistent across views.
        """
        ...

    def check_if_exists(self, key: ImageKey) -> bool:
        """Return ``True`` if the dataset at *key* exists."""
        ...

    def delete_dataset(self, key: ImageKey) -> None:
        """Remove the dataset at *key*.  No-op if it does not exist."""
        ...

    def close(self) -> None:
        """Persist file-level metadata and close the backing store."""
        ...


class HDF5ImageStore:
    """HDF5-backed implementation of ImageDataStore."""

    def __init__(self, path: str | Path) -> None:
        """Open an HDF5 file at *path*, creating it if necessary.

        The parent directory is created automatically.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            self._file = h5py.File(path, mode="r+")
            self._series_count = int(self._file.attrs["series_count"])
            self._channel_count = int(self._file.attrs["channel_count"])
        else:
            self._file = h5py.File(path, mode="w")
            self._series_count = 0
            self._channel_count = 1
            self._file.attrs["series_count"] = self._series_count
            self._file.attrs["channel_count"] = self._channel_count

    @property
    def series_count(self) -> int:
        return self._series_count

    @property
    def channel_count(self) -> int:
        return self._channel_count

    def add_image(
        self,
        key: ImageKey,
        data: np.ndarray,
        angle: float,
        spacing: Sequence[float],
        chunk_size: tuple[int, ...] | None = None,
        calculated: bool | None = None,
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")

        group_path = f"{key.image_type.value}/{key.index}"
        if group_path not in self._file:
            self._file.create_group(group_path)
            if key.image_type == ImageType.ORIGINAL:
                self._series_count += 1
                self._file.attrs["series_count"] = self._series_count

        dataset_name = f"channel_{key.channel}_scale_{key.scale}"
        if dataset_name in self._file[group_path]:
            return

        image_group = self._file[group_path]
        if chunk_size is None:
            image_group.create_dataset(dataset_name, data=data)
        else:
            image_group.create_dataset(dataset_name, data=data, chunks=chunk_size)

        dataset = image_group[dataset_name]
        dataset.attrs["angle"] = float(angle)
        dataset.attrs["spacing"] = list(spacing)
        dataset.attrs["size"] = data.shape

        if calculated is not None:
            dataset.attrs["calculated"] = calculated

    def add_fused_image(
        self,
        channel: int,
        scale: int,
        data: np.ndarray,
        spacing: Sequence[float],
    ) -> None:
        if not isinstance(data, np.ndarray):
            raise TypeError(f"Expected numpy.ndarray, got {type(data).__name__}")

        if "fused" not in self._file:
            image_group = self._file.create_group("fused")
        else:
            image_group = self._file["fused"]

        name = f"channel_{channel}_scale_{scale}"
        if name in image_group:
            return

        image_group.create_dataset(name, data=data)
        image_group[name].attrs["spacing"] = list(spacing)

    def add_transform(
        self,
        index: int,
        channel: int,
        scale: int,
        params: Sequence[float],
        fixed_params: Sequence[float],
        transform_type: int,
    ) -> None:
        path = f"registered/{index}/channel_{channel}_scale_{scale}"
        if path not in self._file:
            raise ValueError(f"Registered dataset {path} does not exist")

        dataset = self._file[path]
        dataset.attrs["tfm_type"] = transform_type
        dataset.attrs["tfm_params"] = list(params)
        dataset.attrs["tfm_fixed_params"] = list(fixed_params)

    def get_image_data(self, key: ImageKey) -> np.ndarray:
        return self._file[key.to_path()][:]

    def get_image_attributes(self, key: ImageKey) -> dict[str, Any]:
        raw = dict(self._file[key.to_path()].attrs)
        result: dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, np.ndarray):
                result[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                result[k] = v.item()
            elif isinstance(v, np.bool_):
                result[k] = bool(v)
            else:
                result[k] = v
        return result

    def get_registered_block(
        self,
        key: ImageKey,
        block_size: np.ndarray,
        block_pad: int,
        block_start_index: np.ndarray,
    ) -> np.ndarray:
        if key.image_type != ImageType.REGISTERED:
            raise ValueError(
                f"get_registered_block requires a REGISTERED key, got {key.image_type}"
            )

        image_size = np.array(self._file[key.to_path()].shape)
        end_index = block_start_index + block_size + block_pad
        start_index = block_start_index - block_pad
        block_idx = arrayutils.start_to_stop_idx(start_index, end_index)

        if (image_size >= end_index).all() and (start_index >= 0).all():
            return self._file[key.to_path()][block_idx]

        pad_block_size = block_size + 2 * block_pad
        block = np.zeros(pad_block_size)

        if (start_index < 0).any():
            block_start = np.negative(start_index.clip(max=0))
            image_start = start_index + block_start
        else:
            block_start = np.zeros(len(block_size), dtype=int)
            image_start = start_index

        if not (image_size >= end_index).all():
            block_crop = end_index - image_size
            block_crop[block_crop < 0] = 0
            block_end = pad_block_size - block_crop
        else:
            block_end = pad_block_size

        end_index = start_index + block_end
        block_read_idx = arrayutils.start_to_stop_idx(image_start, end_index)
        block_write_idx = arrayutils.start_to_stop_idx(block_start, block_end)

        block[block_write_idx] = self._file[key.to_path()][block_read_idx]
        return block

    def get_number_of_images(self, image_type: ImageType) -> int:
        type_str = image_type.value
        if type_str not in self._file:
            return 0
        return len(self._file[type_str])

    def get_scales(self, image_type: ImageType) -> list[int]:
        type_str = image_type.value
        if type_str not in self._file:
            return []

        scales: list[int] = []
        n_images = self.get_number_of_images(image_type)

        for index in range(n_images):
            group_path = f"{type_str}/{index}"
            image_group = self._file[group_path]
            group_scales: list[int] = []
            for name in image_group:
                if name.startswith("channel_"):
                    group_scales.append(int(name.split("_")[-1]))

            if index == 0:
                scales = group_scales
            elif set(scales) != set(group_scales):
                raise ValueError(
                    f"Database inconsistency: scales differ across images "
                    f"for type {type_str}"
                )

        return scales

    def check_if_exists(self, key: ImageKey) -> bool:
        return key.to_path() in self._file

    def delete_dataset(self, key: ImageKey) -> None:
        path = key.to_path()
        if path in self._file:
            del self._file[path]

    def close(self) -> None:
        self._file.attrs["series_count"] = self._series_count
        self._file.attrs["channel_count"] = self._channel_count
        self._file.close()

    def __enter__(self) -> "HDF5ImageStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
