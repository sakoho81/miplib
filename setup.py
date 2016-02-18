from setuptools import setup, find_packages

setup(
    name='supertomo',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'h5py'],
    entry_points={
        'console_scripts': [
            'supertomo.main = supertomo.bin.main:main',
            'supertomo.blurseq = supertomo.bin.convert_files:main'
        ]
    },
    platforms=["any"],
    url='https://bitbucket.org/sakoho81/supertomo2',
    license='BSD',
    author='Sami Koho',
    author_email='sami.koho@gmail.com',
    description='supertomo software was created for Tomographic reconstruction '
                'of STED super-resolution microscopy images.'

)