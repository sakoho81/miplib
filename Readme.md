# SuperTomo v.2

This will be the SuperTomo tomographic reconstructino software reborn. The major changes are:

- adoption of the HDF5 format, which enables working with very large datasets. All the images from a single experiment can also be grouped into a single file.
- CUDA GPU accelerated tomographic fusion. The old fusion method is of course still supported, in case you don't have a compatible GPU.
- SimpleITK. Got rid of the Python wrapped ITK. Now you don't have to compile anything from source.
- Python. I wanted to take advantage of as much as possible of standard Python packages. To this end I tried to get rid of all difficult to maintain/acquire packages, and developed everything for the Anaconda Python distribution.