# MIPLIB

Microscope Image Processing Library (*MIPLIB*) is a Python 2.7 based software package, created especially for processing and analysis of fluorescece microscopy images. It contains functions for example for:

- image registration 2D/3D
- image deconvolution and fusion (2D/3D), based on efficient CUDA GPU accelerated algorithms
<<<<<<< HEAD
- Fourier Ring/Shell Correlation based image resolution analysis -- and several blind image restoration methods based on FRC/FSC.
=======
- Fourier Ring/Shell Correlation (FRC/FSC) based image resolution analysis -- and several blind image restoration methods based on FRC/FSC.
>>>>>>> miplib
- Image quality analysis
- ...

The library is distributed under the following FreeBSD open source license

## How do I install it?

I would recommend going with the *Anaconda* Python distribution, as it removes all the hassle from installing the necessary packages. 

Here's how to setup your machine for development:

 1. There are some C extensions in *miplib* that need to be compiled. Therefore, iff you are on a *Mac*, you will also need to install XCode command line tools. In order to do this, Open *Terminal* and write `xcode-select --install`. In addition, if you already upgraded to MacOS Mojave, you will also have to install the following: `open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg`. If you are on *Windows*, you will need the [C++ compiler](https://www.microsoft.com/en-us/download/details.aspx?id=44266)

<<<<<<< HEAD
3. Open *Terminal* and Clone the *MIBLIB* repository from Bitbucket: `git clone git@bitbucket.org:sakoho81/miplib.git`. The code will go to a sub-directory called *miplib* of the current directory. Put the code somewhere where it can stay for a while.
=======
3. Open *Terminal* and Clone the *MIBLIB* repository from Bitbucket: `git clone git@github.com:sakoho81/miplib.git`. The code will go to a sub-directory called *miplib* of the current directory. Put the code somewhere where it can stay for a while.
>>>>>>> miplib

4. Go to the *miplib* directory and create a new Python virtual environment `conda env create -f environment.yml`. 

5. Activate the created virtual environment by writing `source activate miplib`

6. Now, install the *miplib* package to the new environment by executing the following in the *miplib* directory `python setup.py develop`. This will only create a link to the source code, so don't delete the *miplib* directory afterwards. You can alternatively specify *install* instead of *develop* if you don't want to keep the source code.

<<<<<<< HEAD
If you just want to use MIPLIB functions, you can install it on Anaconda with ...
 
=======
If you just want to use MIPLIB functions, you can install it on Anaconda with ...(available short)

>>>>>>> miplib
## Contribute?

*MIPLIB* was born as a combination of several previously separate libraries. The code and structure, although working, might not in all places make sense. Any suggestions for improvements, new features etc. are welcome. 

## Publications

Here are some works that have been made possible by the MIPLIB (and its predecessors):

Koho, S., T. Deguchi, and P. E. E. Hänninen. 2015. “A Software Tool for Tomographic Axial Superresolution in STED Microscopy.” Journal of Microscopy 260 (2): 208–18.

Koho, Sami, Elnaz Fazeli, John E. Eriksson, and Pekka E. Hänninen. 2016. “Image Quality Ranking Method for Microscopy.” Scientific Reports 6 (July): 28962.

Prabhakar, Neeraj, Markus Peurla, Sami Koho, Takahiro Deguchi, Tuomas Näreoja, H-C Huan-Cheng Chang, Jessica M. J. M. Rosenholm, and Pekka E. P. E. Hänninen. 2017. “STED-TEM Correlative Microscopy Leveraging Nanodiamonds as Intracellular Dual-Contrast Markers.” Small  1701807 (December): 1701807.

Deguchi, Takahiro, Sami Koho, Tuomas Näreoja, and Pekka Hänninen. 2014. “Axial Super-Resolution by Mirror-Reflected Stimulated Emission Depletion Microscopy.” Optical Review 21 (3): 389–94.

Deguchi, Takahiro, Sami V. S. V. Koho, Tuomas Näreoja, Juha Peltonen, and Pekka Hänninen. 2015. “Tomographic STED Microscopy to Study Bone Resorption.” In Proceedings of the SPIE, 9330:93301M – 93301M – 6.


