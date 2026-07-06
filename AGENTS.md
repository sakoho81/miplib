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
│   └── resolution/    # FRC, FSC, analysis
├── bin/               # CLI entry points (wired via pyproject.toml [project.scripts])
├── data/              # Containers, HDF5 I/O, iterators, converters, coordinates
│   ├── containers/    # Image, ArrayDetectorData, FourierCorrelationData, etc.
│   ├── io/            # HDF5 readers/writers for array and FRC data
│   ├── iterators/     # Fourier ring/shell iterators
│   ├── coordinates/   # Polar coordinate grid generators
│   └── core/          # FixedDictionary
├── processing/        # Core algorithms
│   ├── ops_ext.pyx    # Cython extension (compiled to .so)
│   ├── deconvolution/ # Deconvolution (Wiener, RL), CPU + CUDA variants
│   ├── fusion/        # Multi-view fusion, CPU + CUDA variants
│   ├── registration/  # Image registration (ITK-based)
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

## Build & Dev Tooling

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
- Canonical reference: `tests/processing/test_ops_ext.py` (~312 lines, 17 tests).

### Running tests
```
uv run pytest tests/
```

No conftest.py or shared fixtures currently exist. Construct test data inline.

### Markers
- `@pytest.mark.slow` — deselect with `-m "not slow"`
- `@pytest.mark.integration` / `@pytest.mark.unit`

## Test Data Strategy

- **Prefer built-in images**: `skimage.data.camera()`, `skimage.data.shepp_logan_phantom()`, etc. for image processing tests.
- **Custom test data**: Store in `tests/data/`, tracked via Git LFS (`.gitattributes` pattern: `tests/data/** filter=lfs`).
- When `skimage` doesn't provide suitable test data, generate synthetic reference arrays with known properties (e.g. `np.ones`, `np.linspace`, random with fixed seed).
- For Cython extension tests (like `ops_ext`), use small hand-computed arrays to verify correctness.

## Modules Still Untested

The vast majority of modules have zero test coverage. Priority candidates (small, pure logic):

| Priority | Module | Description |
|----------|--------|-------------|
| ✓ done | `tests/processing/test_ops_ext.py` | Cython extension ops |
| ✓ done | `tests/utils/test_numeric.py` | `find_next_power_of_2` |
| ✓ done | `tests/processing/test_converters.py` | deg↔rad conversion |
| medium | `processing/fftutils.py` | FFT wrappers, FFT filters |
| medium | `processing/ndarray.py` | ndarray helper functions |
| medium | `processing/to_string.py` | String formatting utilities |
| medium | `data/coordinates/polar.py` | Polar coordinate grids |
| lower | `processing/deconvolution/*` | Needs image test data |
| lower | `processing/fusion/*` | Needs image test data |
| lower | `processing/registration/*` | Needs SimpleITK |
| lower | `analysis/*` | Depends on data containers |

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
