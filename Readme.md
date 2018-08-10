# MIPLIB

Microscope Image Processing Library (*MIPLIB*) is a Python 2.7 based software package, created especially for processing and analysis of fluorescece microscopy images. It contains functions for example for:

- image registration 2D/3D
- image deconvolution and fusion (2D/3D), based on efficient CUDA GPU accelerated algorithms
- Fourier Ring/Shell Correlation based image resolution analysis -- and several blind image restoration methods based on FRC/FSC.
- Image quality analysis
- ...

The library is distributed under the following FreeBSD open source license
## How do I use it?

I would recommend going with the *Anaconda* Python distribution, as it removes all the hassle from installing the necessary packages. There is a *requirements.txt* file in the repository that can be used to set-up the environment for development. MIPLIB currently only works in Python 2.7, as there are some C extensions that would need to be updated before migrating to Python 3.x. 

Once you have all the necessary packages, simply do ```python setup.py develop```to setup the library for development. I will also try to make conda packages available.

There is a rich command line interace for working with most of the things available in the library. Look at the *bin* directory and *setup.py* to discover more. 

## Contribute?

*MIPLIB* was born as a combination of several previously separate libraries. The code and structure, although working, might not in all places make sense. Any suggestions for improvements, new features etc. are welcome. 

## Publications

Here are some works that have been made possible by the MIPLIB (and its predecessors):





