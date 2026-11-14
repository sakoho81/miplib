from __future__ import annotations

from miplib.data.adapters.image_data import ArrayDataSource, DataSource
from miplib.data.containers.image import Image
from miplib.processing.deconvolution.backends import ViewData, resolve_backend
from miplib.processing.deconvolution.deconvolver import RLDeconvolver, RLOptions
from miplib.processing.deconvolution.estimates import create_estimate
from miplib.processing.deconvolution.psf_utils import prepare_psfs


def richardson_lucy_fusion(
    source: DataSource | list[Image],
    psfs: list[Image],
    *,
    backend: str = "cpu",
    weights: list[float] | None = None,
    backgrounds: list[float] | None = None,
    options: RLOptions | None = None,
) -> Image:
    """Multi-view Richardson-Lucy deconvolution / fusion.

    Parameters
    ----------
    source : DataSource | list[Image]
        One or more views of the same sample.  A list of Images is
        wrapped in an ArrayDataSource automatically.
    psfs : list[Image]
        One PSF per view.
    backend : {"cpu", "cuda"}
        Computation backend.
    weights : list[float] | None
        View weights.  Defaults to ``[1.0, ...]``.
    backgrounds : list[float] | None
        Per-view background estimates.  Defaults to ``[0.0, ...]``.
    options : RLOptions | None
        Full algorithm options.  Uses ``RLOptions()`` defaults when
        not given (100 iterations, IMAGE_MEAN first estimate, etc.).

    Returns
    -------
    Image
        Fused / deconvolved image.
    """
    if isinstance(source, list):
        source = ArrayDataSource(source)

    n = source.n_views
    if len(psfs) != n:
        raise ValueError(f"Expected {n} PSF(s), got {len(psfs)}")

    opts = options or RLOptions()
    weights = weights or [1.0] * n
    backgrounds = backgrounds or [0.0] * n

    norms, adjs = prepare_psfs(psfs, source.spacing)
    vd = ViewData(source, norms, adjs, weights, backgrounds)
    backend_inst = resolve_backend(backend, vd, source.shape)
    estimate = create_estimate(source, strategy=opts.estimate_strategy)
    return RLDeconvolver(backend_inst, estimate, options=opts).run()


def richardson_lucy_deconvolution(
    image: Image,
    psf: Image,
    *,
    backend: str = "cpu",
    options: RLOptions | None = None,
) -> Image:
    """Single-image Richardson-Lucy deconvolution.

    Thin wrapper around :func:`richardson_lucy_fusion` for the common
    case of one image and one PSF.

    Parameters
    ----------
    image : Image
        Observed / blurred image.
    psf : Image
        Point-spread function.
    backend : {"cpu", "cuda"}
        Computation backend.
    options : RLOptions | None
        Full algorithm options.  Uses ``RLOptions()`` defaults when
        not given (100 iterations, IMAGE_MEAN first estimate, etc.).

    Returns
    -------
    Image
        Deconvolved image.
    """
    return richardson_lucy_fusion(
        source=[image],
        psfs=[psf],
        backend=backend,
        options=options,
        weights=[1.0],
        backgrounds=[0.0],
    )
