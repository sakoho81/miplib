# MIPLIB
[![DOI](https://zenodo.org/badge/162555135.svg)](https://zenodo.org/badge/latestdoi/162555135)

Microscope Image Processing Library (*MIPLIB*) is a Python based software library, created especially for processing and analysis of fluorescece microscopy images. It contains functions for example for:

- image registration 2D/3D
- image deconvolution and fusion (2D/3D), based on efficient CUDA GPU accelerated algorithms
- Fourier Ring/Shell Correlation (FRC/FSC) based image resolution analysis -- and several blind image restoration methods based on FRC/FSC.
- Image quality analysis
- ...

The library is distributed under a BSD open source license.

## Installation

MIPLIB uses modern Python packaging with [uv](https://docs.astral.sh/uv/) for fast, reliable dependency management. The library requires **Python 3.11+** and works on all platforms (Windows, macOS, Linux).

### Prerequisites

1. **Python 3.11+**: MIPLIB requires a modern Python version
2. **C Compiler**: For compiling Cython extensions
   - **macOS**: Install Xcode command line tools: `xcode-select --install`
   - **Windows**: Install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
   - **Linux**: Install `build-essential` (Ubuntu/Debian) or equivalent
3. **Java Runtime (Optional)**: Required only if using the Bioformats reader for microscopy formats
   - Set `JAVA_HOME` environment variable
   - See [JPype installation guide](https://jpype.readthedocs.io/en/latest/install.html) for details

### Quick Installation

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   # Or on Windows: powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/sakoho81/miplib.git
   cd miplib
   ```

3. **Install MIPLIB** (creates virtual environment automatically):
   ```bash
   uv sync
   ```

4. **Activate the environment**:
   ```bash
   source .venv/bin/activate  # On Unix/macOS
   # Or on Windows: .venv\Scripts\activate
   ```

### Development Setup

For contributors and developers:

1. **Fork and clone** the repository:
   ```bash
   git clone git@github.com:<your_account>/miplib.git
   cd miplib
   ```

2. **Install with development dependencies**:
   ```bash
   uv sync --extra dev
   ```

3. **Install pre-commit hooks** (recommended):
   ```bash
   source .venv/bin/activate
   pre-commit install
   ```

4. **Run tests** to verify everything works:
   ```bash
   source .venv/bin/activate
   pytest tests/
   ```

### Optional GPU Support

For CUDA acceleration (NVIDIA GPUs):
```bash
uv sync --extra cuda
```

### Installation from PyPI (Coming Soon)

Once published to PyPI:
```bash
uv add miplib
# Or with pip: pip install miplib
```

## How do I use it?

My preferred tool for explorative tasks is Jupyter Notebook/Lab. Please look for updates in the notebooks/ folder (a work in progress). Let me know if you would be interested in some specific example to be included.

There are also a number of command line scripts (entry points) in the bin/ directory that may be handy in different batch processing tasks. They are also a good place to start exploring the library.

## Contribute?

*MIPLIB* was born as a combination of several previously separate libraries. The code and structure, although working, might (does) not in all places make sense. Any suggestions for improvements, new features etc. are welcome.

## About GPU acceleration

The deconvolution algorithms can be accelerated with a GPU. On MacOS the CUDA GPU acceleration currently does not work, because there are no NVIDIA drivers available for the latest OS versions. I recently re-factored the GPU acceleration functions, using the CuPy library. It would in principle be possible to use OpenCL backend, instead of CUDA, but I have not tried that (yet).

## Publications

Here are some works that have been made possible by the MIPLIB (and its predecessors):

Koho, S. V. et al. Two-photon image-scanning microscopy with SPAD array and blind image reconstruction. Biomed. Opt. Express, BOE 11, 2905–2924 (2020)

[Koho, S. *et al.* Fourier ring correlation simplifies image restoration in fluorescence microscopy. Nat. Commun. 10 3103 (2019).](https://doi.org/10.1038/s41467-019-11024-z)

Koho, S., T. Deguchi, and P. E. E. Hänninen. 2015. “A Software Tool for Tomographic Axial Superresolution in STED Microscopy.” Journal of Microscopy 260 (2): 208–18.

Koho, Sami, Elnaz Fazeli, John E. Eriksson, and Pekka E. Hänninen. 2016. “Image Quality Ranking Method for Microscopy.” Scientific Reports 6 (July): 28962.

Prabhakar, Neeraj, Markus Peurla, Sami Koho, Takahiro Deguchi, Tuomas Näreoja, H-C Huan-Cheng Chang, Jessica M. J. M. Rosenholm, and Pekka E. P. E. Hänninen. 2017. “STED-TEM Correlative Microscopy Leveraging Nanodiamonds as Intracellular Dual-Contrast Markers.” Small  1701807 (December): 1701807.

Deguchi, Takahiro, Sami Koho, Tuomas Näreoja, and Pekka Hänninen. 2014. “Axial Super-Resolution by Mirror-Reflected Stimulated Emission Depletion Microscopy.” Optical Review 21 (3): 389–94.

Deguchi, Takahiro, Sami V. Koho, Tuomas Näreoja, Juha Peltonen, and Pekka Hänninen. 2015. “Tomographic STED Microscopy to Study Bone Resorption.” In Proceedings of the SPIE, 9330:93301M – 93301M – 6.
