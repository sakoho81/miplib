from setuptools import setup, find_packages, Extension
import numpy

setup(
    name='supertomo',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'h5py', 'SuperTomo', 'SimpleITK',
                      'matplotlib', 'accelerate', 'numba'],
    entry_points={
        'console_scripts': [
            'supertomo.import = supertomo.bin.import:main',
            'supertomo.correlatem = supertomo.bin.correlatem:main',
            'supertomo.transform = supertomo.bin.transform:main',
            'supertomo.fuse = supertomo.bin.fuse:main',
            'supertomo.register = supertomo.bin.register:main'
        ]
    },
    platforms=["any"],
    url='https://bitbucket.org/sakoho81/supertomo2',
    license='BSD',
    author='Sami Koho',
    author_email='sami.koho@gmail.com',
    description='supertomo software was created for Tomographic reconstruction '
                'of STED super-resolution microscopy images.',
    ext_modules=[
        Extension(
            'supertomo.reconstruction.ops_ext',
            ['supertomo/reconstruction/src/ops_ext.c'],
            include_dirs=[numpy.get_include()]),
        Extension(
            'supertomo.io._tifffile',
            ['supertomo/io/src/tifffile.c'],
            include_dirs=[numpy.get_include()])
    ]
)