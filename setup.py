import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# Define Cython extensions with proper include directories
extensions = [
    Extension(
        "miplib.processing.ops_ext",
        ["miplib/processing/ops_ext.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

ext_modules = cythonize(extensions, compiler_directives={"language_level": 3})

setup(ext_modules=ext_modules)
