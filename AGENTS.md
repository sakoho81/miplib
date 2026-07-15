# AGENTS.md

## Project Overview

miplib is a Python library for (optical) microscopy image restoration, reconstruction, and analysis.
- Python >=3.11, built with setuptools + Cython
- Key dependencies: numpy, scipy, scikit-image, SimpleITK, numba, h5py, matplotlib

## Codebase Structure

```
miplib/
├── analysis/          # Image quality ranking, FRC/FSC resolution analysis
│   ├── image_quality/ # Filters, quality ranking, utils
│   └── resolution/    # FRC, FSC, analysis, common (shared utilities)
├── bin/               # CLI entry points (wired via pyproject.toml [project.scripts])
├── data/              # Containers, HDF5 I/O, iterators, converters, coordinates, adapters
│   ├── containers/    # Image, ArrayDetectorData, FourierCorrelationData, etc.
│   ├── adapters/      # DataSource protocols and adapters (registration, deconvolution, array detector)
│   ├── io/            # HDF5 readers/writers for array and FRC data
│   ├── iterators/     # Fourier ring/shell iterators
│   ├── coordinates/   # Polar coordinate grid generators
│   └── core/          # FixedDictionary
├── processing/        # Core algorithms
│   ├── ops_ext.pyx    # Cython extension (compiled to .so)
│   ├── deconvolution/ # Deconvolution (Wiener, RL), CPU + CUDA variants
│   ├── registration/  # Registration (ITK, phase correlation), MultiViewRegistration
│   ├── ism/           # Image Scanning Microscopy reconstruction
│   ├── segmentation/  # Masking
│   ├── fftutils.py    # FFT/IFFT wrappers, FFT filters
│   ├── ndarray.py     # ndarray helpers
│   ├── transform.py   # Coordinate transforms
│   ├── windowing.py   # Hamming/Tukey windowing
│   ├── image.py       # Image container
│   ├── converters.py  # Degrees/radians
│   └── to_string.py   # String formatting utilities
├── psf/               # PSF generation (psfgen)
├── ui/                # CLI arg parsing (argparse helpers), matplotlib plots
└── utils/             # Small helpers: generic, numeric, string
```

## Branch & PR Discipline

- **NEVER commit directly to `public`.** Always create a feature branch and open a PR.
- **Never merge a PR.** The user handles merges.
- Branch naming: `feature/<name>`, `refactor/<name>`, `fix/<name>`, `test/<name>`.

- **Package manager**: `uv` — use `uv sync --group dev` to install all deps
- **Linting/formatting**: `ruff` (line-length 88, double quotes, isort)
- **pre-commit**: CI runs `pre-commit run --all-files`
- **mypy**: Runs on pre-push (via pre-commit hook). Parameters: `--ignore-missing-imports --no-strict-optional`. Excludes `tests/` and `setup.py`.
- **Cython extension**: `miplib/processing/ops_ext.pyx` compiled via `setup.py`, outputs `.so` files in `miplib/processing/`
- **CI**: `.github/workflows/ci.yml` — runs pre-commit and `pytest tests/`

## Testing Conventions

### Location & discovery
- All tests live under the root `tests/` directory, mirroring the `miplib/` package structure.
  - `miplib/utils/numeric.py` → `tests/utils/test_numeric.py`
  - `miplib/processing/converters.py` → `tests/processing/test_converters.py`
- `pytest.ini` only discovers from `tests/` (`testpaths = tests`).
- Inline test directories inside `miplib/` (e.g. `data/core/tests/`, `data/io/tests/`) are **not discovered** by default. If extending those tests, migrate them into the root `tests/` directory.

### Style
- **Flat functions**: `def test_<name>()` — no `unittest.TestCase` subclassing.
- **Parametrize** for dtype or shape variants: `@pytest.mark.parametrize`.
- **Array assertions** via `numpy.testing` (`assert_array_almost_equal`, `assert_almost_equal`).
- **Exception testing** via `pytest.raises(...)` with `match=` for message validation.
- **No tautologies** — every assertion must verify actual behavioral properties of the code under test.
- **No repetition** — don't test the same logic through multiple redundant cases.
- **Test meaningful behaviour, not just "it runs."** The best tests verify deterministic signal-processing properties using test patterns with known characteristics (Gaussian self-duality, spectral leakage reduction from windowing, step-edge ringing/overshoot from hard frequency cutoffs, constant-image DC identity). Avoid tests whose only assertion is that the output has the correct shape or dtype — those can be folded into a more substantive test that also checks a genuine property, or dropped if the property is already covered elsewhere.
- Canonical reference: `tests/processing/test_ops_ext.py` (~312 lines, 17 tests).

### Running tests
```
uv run pytest tests/
```

Conftest.py with shared Image fixtures and pattern generators lives at `tests/conftest.py`.

### Markers
- `@pytest.mark.slow` — deselect with `-m "not slow"`
- `@pytest.mark.integration` / `@pytest.mark.unit`

## Test Data Strategy

- **Prefer built-in images**: `skimage.data.camera()`, `skimage.data.shepp_logan_phantom()`, `skimage.data.binary_blobs(n_dim=3)` for image processing tests.
- **Shared fixtures** are in `tests/conftest.py`: `image_2d`, `image_3d`, `gaussian_2d`, `camera_image`, `shepp_logan`, `blobs_3d`, `psf_gaussian_2d`, `frc_options`.
- **Pattern generators** in `tests/conftest.py`: `gaussian_spot(shape, sigma)`, `sine_grating(shape, frequency)`, `impulse(shape)`, `step_edge(shape, axis)`, `bin_aligned_sine(shape, n_cycles, axis)`, `two_frequency_signal(shape, low, high, axis)`, `checkerboard_pattern(shape)`.
- **When writing new tests**, prefer conftest fixtures and pattern generators over local duplicates. If a test creates a reusable deterministic pattern (e.g. a 0/1 checkerboard), add it to conftest as a pattern generator function so other test modules can share it. Keep fixtures and generators documented in the lists above.
- When a new Image-based pattern is needed, import pattern generators from conftest via `from tests.conftest import <name>` and wrap with `Image(...)` inline.
- **Custom test data**: Store in `tests/testdata/`, tracked via Git LFS for binary files (`.hdf5`, `.tif`, `.mat`). Python source files in `tests/testdata/` are regular git.
- When `skimage` doesn't provide suitable test data, generate synthetic reference arrays with known properties (e.g. `np.ones`, `np.linspace`, random with fixed seed).
- For Cython extension tests (like `ops_ext`), use small hand-computed arrays to verify correctness.

## Modules Still Untested

The vast majority of modules have zero test coverage. Priority candidates (small, pure logic):

| Priority | Module | Description |
|----------|--------|-------------|
| ✓ done | `tests/processing/test_ops_ext.py` | Cython extension ops |
| ✓ done | `tests/utils/test_numeric.py` | `find_next_power_of_2` |
| ✓ done | `tests/processing/test_converters.py` | deg↔rad conversion |
| ✓ done | `tests/processing/test_ndarray.py` | ndarray helper functions |
| ✓ done | `tests/processing/test_to_string.py` | String formatting utilities |
| ✓ done | `tests/ui/test_progress.py` | Terminal progress bar |
| ✓ done | `tests/data/containers/test_image.py` | Image (ndarray subclass + spacing) |
| ✓ done | `tests/data/containers/test_array_detector_data.py` | ArrayDetectorData container |
| ✓ done | `tests/data/coordinates/test_polar.py` | Polar coordinate grids |
| ✓ done | `tests/processing/test_fftutils.py` | FFT wrappers, FFT filters (now has Image fixtures) |
| ✓ done | `tests/psf/test_psfgen.py` | PSF generation from FWHM |
| ✓ done | `tests/data/core/test_dictionary.py` | FixedDictionary (immutable-key dict) |
| ✓ done | `tests/data/containers/test_fourier_correlation_data.py` | FRC/FSC data containers |
| ✓ done | `tests/processing/test_image.py` | Image operations, translation, checkerboards, noise, contrast (63 tests) |
| ✓ done | `tests/processing/test_registration.py` | Registration: phase correlation, ITK convergence, multi-view, data sources (34 tests) |
| ✓ done | `tests/test_register_cli.py` | Register CLI: parsing, source resolution, options (11 tests) |
| lower | `processing/deconvolution/*` | Now has Image + blobs_3d + psf_gaussian_2d |
| lower | `processing/fusion/*` | Now has Image + blobs_3d fixtures |
| ✓ done | `tests/analysis/resolution/` | FRCOptions, accumulate, curve builder, analysis, first_guess, 3D iterator, sectioned accumulation |
| ✓ done | `tests/psf/test_psfgen.py` | FRC-based PSF generation (real ISM image) |
| ✓ done | `tests/processing/test_deconvolver.py` | RL deconv pipeline with FRC tracker (real ISM image) |
| ✓ done | `tests/data/core/test_dictionary.py` | FixedDictionary — __contains__, __iter__, unset-keys-return-None |

## Known Gaps / Future Work

- **CLI migration to dataclass + `--options` JSON pattern** — The `bin/pyimq.py`, `bin/power.py`, and `bin/resolution.py` now use a typed `@dataclass` options class + `options_from_dict()` bridge (via dacite). The deconvolution CLI (`bin/deconvolve.py`) has partial support via `--frc-options` and `--tracker-type`. Remaining CLIs (`ism`, `registration`, `fuse`) still build bare `argparse.Namespace` objects and pass them as untyped bags of attributes. Migrating them to the same pattern would:
  - Make library functions accept typed dataclasses (`RLOptions`, `RegistrationOptions`, `FusionOptions`)
  - Give non-CLI callers a clean programmatic API without constructing `argparse.Namespace`
  - Allow flexible CLI usage via `--options` JSON override (useful for automated/agent-driven workflows)
- **FFT primitives consolidation** — `miplib/processing/deconvolution/backends.py` defines `_CPUFFT`/`_CUDAFFT`/`resolve_fft` for GPU-aware `fftn`/`ifftn` dispatch. These should move into `miplib/processing/fftutils.py` so the whole library has one CPU+CUDA FFT layer. `fftutils.fft()` and `ifft()` would gain an optional `backend` kwarg. Currently `backends.py` and `wiener.py` each do their own `cupy` import guard — this should become a single import in `fftutils`.
- **Estimate checkpoint support** — Saving/loading estimate state mid-deconvolution would allow resuming from checkpoints. Needs `EstimateIO.save(estimate, path)` / `load(path, shape, spacing)` plus metadata (iteration count, tracker state, PSF parameters).

## Code Style

- Python >=3.11, no legacy compat needed
- Line length 88 (ruff), double quotes, isort import ordering
- Cython files (`.pyx`) use `language_level=3`

### Type annotations
- Add type annotations to all function signatures when touching a module
- Prefer concise docstrings (one-line summary) over `:param`/`:type`/`:rtype` directives — type annotations carry that information

### Logging
- Use `logging` (`logging.getLogger(__name__)`) instead of `print()` in library code
- `print()` is acceptable only in CLI entry points (`miplib/bin/`)

### Assertions vs. type checks
- Prefer static type checking (mypy) over runtime assertions for type validation
- Prefer static type checking alone — delete runtime `isinstance`/`issubclass` checks unless the function is a public API boundary where static analysis can't protect callers
- Avoid bare `assert` for input validation in library functions — raise `TypeError`/`ValueError` explicitly (assertions can be disabled with `python -O`)

### When writing tests for a module
- Fix obvious code quality issues alongside: missing type annotations, outdated docstring style, `print()` → `logging`, assertion anti-patterns
