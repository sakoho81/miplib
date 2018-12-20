from setuptools import setup, find_packages, Extension
import numpy

setup(
    name='miplib',
    version='1.0',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'h5py', 'SimpleITK',
                      'matplotlib', 'numba', 'pyculib', 'pandas', 'pims'],
    entry_points={
        'console_scripts': [
            'miplib.import = miplib.bin.import:main',
            'miplib.correlatem = miplib.bin.correlatem:main',
            'miplib.transform = miplib.bin.transform:main',
            'miplib.fuse = miplib.bin.fuse:main',
            'miplib.register = miplib.bin.register:main',
            'miplib.deconvolve = miplib.bin.deconvolve:main',
            'miplib.resolution = miplib.bin.resolution:main',
            'miplib.3dfrc = miplib.bin.threedfrc:main',
            'pyimq.main = miplib.bin.pyimq:main',
            'pyimq.util.blurseq = miplib.bin.utils.create_blur_sequence:main',
            'pyimq.util.imseq = miplib.bin.utils.create_photo_test_set:main',
            'pyimq.subjective = miplib.bin.subjective:main',
            'pyimq.power = miplib.bin.power:main'
        ]
    },
    platforms=["any"],
    url='https://bitbucket.org/sakoho81/miplib',
    license='BSD',
    author='Sami Koho',
    author_email='sami.koho@gmail.com',
    description='miplib software was created for Tomographic processing '
                'of STED super-resolution microscopy images.',
    ext_modules=[
        Extension(
            'miplib.processing.ops_ext',
            ['miplib/processing/src/ops_ext.c'],
            include_dirs=[numpy.get_include()]),
        Extension(
            'miplib.data.io._tifffile',
            ['miplib/data/io/src/tifffile.c'],
            include_dirs=[numpy.get_include()]),
        Extension(
            'miplib.psf._psf',
            ['miplib/psf/src/psf.c'],
            include_dirs=[numpy.get_include()]),
    ]
)