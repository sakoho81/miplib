"""Data source adapters for registration.

Provides lazy-loading interfaces for accessing multi-view image data,
supporting both in-memory arrays and HDF5-backed storage.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import SimpleITK as sitk

from miplib.data.containers.array_detector_data import ArrayDetectorData
from miplib.data.containers.image import Image
from miplib.data.containers.image_data import ImageData, ImageKey, ImageType
from miplib.processing.itk import get_itk_transform_parameters


class RegistrationDataSource(Protocol):
    """Protocol for lazy-loading multi-view image data.

    Implementations provide on-demand access to images and can
    optionally persist registration results.
    """

    n_views: int
    spacing: tuple[float, ...]

    def get_image(self, view: int) -> Image:
        """Load image for the given view index."""
        ...

    def exists(self, view: int) -> bool:
        """Check if a registered result already exists for this view."""
        ...

    def save_result(self, view: int, image: Image, transform: sitk.Transform) -> None:
        """Save registration result (resampled image and transform)."""
        ...


class ArrayRegistrationDataSource(RegistrationDataSource):
    """In-memory data source wrapping a list of Images.

    Suitable for small datasets or testing.  When *output_dir* or
    *output_paths* is given, ``save_result`` writes the resampled image
    to disk.

    Parameters
    ----------
    images : list of Image
        The images to register.
    output_dir : str | None
        Directory for auto-named output files (``registered_{view}.tif``).
    output_paths : list of str | None
        Explicit per-view output paths (takes precedence over *output_dir*).
    """

    def __init__(
        self,
        images: list[Image],
        output_dir: str | None = None,
        output_paths: list[str] | None = None,
    ) -> None:
        if not images:
            raise ValueError("images list must not be empty")
        self._images = images
        self._output_dir = output_dir
        self._output_paths = output_paths
        self.n_views = len(images)
        self.spacing = tuple(images[0].spacing)

    def get_image(self, view: int) -> Image:
        return self._images[view]

    def exists(self, view: int) -> bool:
        return False

    def save_result(self, view: int, image: Image, transform: sitk.Transform) -> None:
        path = None
        if self._output_paths and view < len(self._output_paths):
            path = self._output_paths[view]
        elif self._output_dir:
            path = str(Path(self._output_dir) / f"registered_{view}.tif")

        if path:
            from miplib.data.io.write import image as write_image

            write_image(path, image)


class ArrayDetectorDataSource(RegistrationDataSource):
    """Data source wrapping an ``ArrayDetectorData`` container.

    Provides a view over the 2D detector array without copying images.
    """

    def __init__(self, data: ArrayDetectorData, photosensor: int = 0) -> None:
        if photosensor >= data.ngates:
            raise ValueError(
                f"photosensor={photosensor} out of range (ngates={data.ngates})"
            )

        self._data = data
        self._photosensor = photosensor
        self.n_views = data.ndetectors
        self.spacing = tuple(data[photosensor, 0].spacing)

    def get_image(self, view: int) -> Image:
        return self._data[self._photosensor, view]

    def exists(self, view: int) -> bool:
        return False

    def save_result(self, view: int, image: Image, transform: sitk.Transform) -> None:
        pass


class HDF5RegistrationDataSource(RegistrationDataSource):
    """HDF5-backed data source with lazy loading.

    Reads images one at a time from an ImageData container, never loading
    all views into memory simultaneously. Always reads from ORIGINAL and
    writes to REGISTERED.
    """

    def __init__(
        self,
        data: ImageData,
        views: list[int],
        channel: int = 0,
        scale: int = 100,
    ) -> None:
        if not views:
            raise ValueError("views list must not be empty")

        self._data = data
        self._views = list(views)
        self._channel = channel
        self._scale = scale
        self.n_views = len(views)

        key = ImageKey(ImageType.ORIGINAL, views[0], channel, scale)
        self.spacing = tuple(data.get_voxel_size(key))

    def get_image(self, view: int) -> Image:
        key = ImageKey(
            ImageType.ORIGINAL, self._views[view], self._channel, self._scale
        )
        return self._data.get_image(key)

    def exists(self, view: int) -> bool:
        return self._data.check_if_exists(
            ImageType.REGISTERED, self._views[view], self._channel, self._scale
        )

    def save_result(self, view: int, image: Image, transform: sitk.Transform) -> None:
        self._data.add_registered_image(
            image, self._scale, self._views[view], self._channel, 0, self.spacing
        )
        tfm_name, params, fixed_params = get_itk_transform_parameters(transform)
        self._data.add_transform(
            self._scale,
            self._views[view],
            self._channel,
            params,
            fixed_params,
            tfm_name,  # type: ignore[arg-type]
        )
