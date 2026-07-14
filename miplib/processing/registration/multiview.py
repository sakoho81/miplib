"""Multi-view image registration.

Registers a collection of images to a chosen reference using any
``RegistrationBackend``.
"""

from __future__ import annotations

import logging
from typing import Iterator

import SimpleITK as sitk

from miplib.data.adapters.registration_data import RegistrationDataSource

from .methods import RegistrationBackend, create_backend
from .options import RegistrationMethod, RegistrationOptions

logger = logging.getLogger(__name__)


class MultiViewRegistration:
    """Register multiple images from a data source to a common reference.

    Images are loaded one at a time via the data source, never holding
    more than the reference and one moving image in memory.

    The class is iterable — each iteration step registers the next view
    and yields the resulting transform::

        mv = MultiViewRegistration(source)
        for transform in mv:
            print(transform.GetParameters())

    Call ``execute()`` for a plain list of transforms.
    """

    def __init__(
        self,
        source: RegistrationDataSource,
        method: RegistrationMethod = RegistrationMethod.ITERATIVE_RIGID,
        fixed_idx: int = 0,
        options: RegistrationOptions | None = None,
    ) -> None:
        if fixed_idx >= source.n_views:
            raise ValueError(
                f"fixed_idx={fixed_idx} out of range (n_views={source.n_views})"
            )
        self.source = source
        self.fixed_idx = fixed_idx
        self.options = options or RegistrationOptions(method=method)
        self.transforms: list[sitk.Transform] = []
        self._backend: RegistrationBackend | None = None

    def _get_backend(self) -> RegistrationBackend:
        if self._backend is None:
            self._backend = create_backend(self.options)
        return self._backend

    def execute(self) -> list[sitk.Transform]:
        """Register all images sequentially. Returns list of transforms."""
        return list(self)

    def __iter__(self) -> Iterator[sitk.Transform]:
        self.transforms = []
        backend = self._get_backend()

        # Load reference image once
        fixed = self.source.get_image(self.fixed_idx)
        ndim = fixed.ndim

        for idx in range(self.source.n_views):
            if idx == self.fixed_idx:
                identity = sitk.TranslationTransform(ndim)
                self.transforms.append(identity)
                yield identity
                continue

            logger.info("Registering view %d to view %d", idx, self.fixed_idx)
            moving = self.source.get_image(idx)
            transform = backend.register(fixed, moving)
            self.transforms.append(transform)
            yield transform

        logger.info("Multi-view registration complete")
