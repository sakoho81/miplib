# MIPLIB

Microscope Image Processing Library (*MIPLIB*) is a Python based software library, created especially for processing and analysis of fluorescece microscopy images. It contains functions for example for:

- image registration 2D/3D
- image deconvolution and fusion (2D/3D), based on efficient CUDA GPU accelerated algorithms
- Fourier Ring/Shell Correlation (FRC/FSC) based image resolution analysis -- and several blind image restoration methods based on FRC/FSC.
- Image quality analysis
- image scanning microscopy (ISM) image reconstruction
- ...

The library is distributed under a BSD open source license.

## How do I install it?

I would recommend going with the *Anaconda* Python distribution, as it removes all the hassle from installing the necessary packages. MIPLIB should work on all platforms (Windows, MacOS, Linux), however I do not actively test it on Windows. On MacOS the CUDA GPU acceleration currently does not work, because there are no NVIDIA drivers available for the latest OS versions.

### Here's how to setup your machine for development:

  1. There are some C extensions in *miplib* that need to be compiled. Therefore, if you are on a *mac*, you will also need to install XCode command line tools. In order to do this, Open *Terminal* and write `xcode-select --install`. In addition, if you already upgraded to MacOS Mojave, you will also have to install the following: `open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg`. If you are on *Windows*, you will need the [C++ compiler](https://wiki.python.org/moin/WindowsCompilers)


3. Clone the *MIBLIB* repository from GitHub (`git clone git@github.com:sakoho81/miplib.git`). The code will be saved to a sub-directory called *miplib* of the current directory. Put the code somewhere where it can stay for a while. You may need to generate an SSH key, if you have not used GitHub previously.

4. Go to the *miplib* directory and create a new Python virtual environment `conda env create -f environment.yml`. 

5. Activate the created virtual environment by writing `conda activate miplib`

6. Now, install the *miplib* package to the new environment by executing the following in the *miplib* directory `python setup.py develop`. This will only create a link to the source code, so don't delete the *miplib* directory afterwards. 

### And if you are not a developer

If you just want to use the library, you can get everything running as follows:

1. Download the *environment_client.yml* file and create a Python virtual environment `conda env create -f environment_client.yml`. 

2. Activate the created virtual environment by writing `conda activate miplib`

## How do I use it?

My preferred tool for explorative tasks is Jupyter Notebook/Lab. Please look for updates in the Examples/ folder (a work in progress). Let me know if you would be interested in some specific example to be included. 

There are also a number of command line scripts (entry points) in the bin/ directory that may be handy in different batch processing tasks. However, some of them may require some work, as I haven't really used the command line interface in a while.

## Contribute?

*MIPLIB* was born as a combination of several previously separate libraries. The code and structure, although working, might (does) not in all places make sense. Any suggestions for improvements, new features etc. are welcome. 

## Regarding Python versions

I recenly migrated MIPLIB to Python 3, and have no intention to maintain backwards compatibility to Python 2.7. You can checkout an older version of the library, if you need to work on Python 2.7.

## Publications

Here are some works that have been made possible by the MIPLIB (and its predecessors):


[Koho, S. *et al.* Fourier ring correlation simplifies image restoration in fluorescence microscopy. Nat. Commun. 10 3103 (2019).](https://doi.org/10.1038/s41467-019-11024-z)

Koho, S. V. *et al.* Easy Two-Photon Image Scanning Microscopy with SPAD Array and Blind Image Reconstruction. *biorxiv* doi:10.1101/563288

Koho, S., T. Deguchi, and P. E. E. Hänninen. 2015. “A Software Tool for Tomographic Axial Superresolution in STED Microscopy.” Journal of Microscopy 260 (2): 208–18.

Koho, Sami, Elnaz Fazeli, John E. Eriksson, and Pekka E. Hänninen. 2016. “Image Quality Ranking Method for Microscopy.” Scientific Reports 6 (July): 28962.

Prabhakar, Neeraj, Markus Peurla, Sami Koho, Takahiro Deguchi, Tuomas Näreoja, H-C Huan-Cheng Chang, Jessica M. J. M. Rosenholm, and Pekka E. P. E. Hänninen. 2017. “STED-TEM Correlative Microscopy Leveraging Nanodiamonds as Intracellular Dual-Contrast Markers.” Small  1701807 (December): 1701807.

Deguchi, Takahiro, Sami Koho, Tuomas Näreoja, and Pekka Hänninen. 2014. “Axial Super-Resolution by Mirror-Reflected Stimulated Emission Depletion Microscopy.” Optical Review 21 (3): 389–94.

Deguchi, Takahiro, Sami V. Koho, Tuomas Näreoja, Juha Peltonen, and Pekka Hänninen. 2015. “Tomographic STED Microscopy to Study Bone Resorption.” In Proceedings of the SPIE, 9330:93301M – 93301M – 6.

