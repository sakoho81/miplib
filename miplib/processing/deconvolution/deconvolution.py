"""High-level Richardson-Lucy deconvolution and multi-view fusion."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import ViewData, resolve_backend
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
from miplib.processing.deconvolution.estimates import create_estimate
from miplib.processing.deconvolution.psf_utils import prepare_psfs

if TYPE_CHECKING:
    from miplib.data.adapters.image_data import DataSource

IterationCallback = Callable[[RLDeconvolver], None]


def richardson_lucy_fusion(
    source: DataSource | list[Image],
    psfs: list[Image],
    *,
    backend: str = "cpu",
    weights: list[float] | None = None,
    backgrounds: list[float] | None = None,
    options: RLOptions | None = None,
    estimate_path: str | Path | None = None,
    on_iteration: IterationCallback | None = None,
) -> Image:
    """Run multi-view Richardson-Lucy deconvolution.

    ``source`` is a block-wise :class:`DataSource` or a list of
    :class:`Image` views, one PSF per view. ``estimate_path`` backs the
    estimate with a memory-mapped file. ``on_iteration`` is called with the
    running :class:`RLDeconvolver` after each iteration.
    """
    if isinstance(source, list):
        from miplib.data.adapters.image_data import ArrayDataSource

        source = ArrayDataSource(source)

    n_views = source.n_views
    if len(psfs) != n_views:
        raise ValueError(f"Expected {n_views} PSF(s), got {len(psfs)}")

    opts = options or RLOptions()
    weights = weights or [1.0] * n_views
    backgrounds = backgrounds or [0.0] * n_views

    norms, adjs = prepare_psfs(psfs, source.spacing)
    view_data = ViewData(source, norms, adjs, weights, backgrounds)
    backend_inst = resolve_backend(backend, view_data, source.shape)
    estimate = create_estimate(
        source,
        opts.first_estimate,
        constant=opts.estimate_constant,
        path=estimate_path,
    )

    deconv = RLDeconvolver(backend_inst, estimate, options=opts)
    if on_iteration is None:
        return deconv.run()
    for _ in deconv:
        on_iteration(deconv)
    return deconv.result()


def richardson_lucy_deconvolution(
    image: Image,
    psf: Image,
    *,
    backend: str = "cpu",
    options: RLOptions | None = None,
    estimate_path: str | Path | None = None,
    on_iteration: IterationCallback | None = None,
) -> Image:
    """Deconvolve a single image with a single PSF."""
    return richardson_lucy_fusion(
        source=[image],
        psfs=[psf],
        backend=backend,
        weights=[1.0],
        backgrounds=[0.0],
        options=options,
        estimate_path=estimate_path,
        on_iteration=on_iteration,
    )
