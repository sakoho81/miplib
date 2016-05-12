from setuptools import setup, find_packages, Extension
import numpy

setup(
    name='supertomo',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'h5py', 'SimpleITK'],
    entry_points={
        'console_scripts': [
            'supertomo.main = supertomo.bin.main:main',
            'supertomo.import = supertomo.bin.import:main',
            'supertomo.test.hdf5 = supertomo.bin.test.test_hdf5.py:main'
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