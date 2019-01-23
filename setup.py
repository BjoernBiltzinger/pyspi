from setuptools import setup
#from distutils.core import setup

import os

# Create list of data files
def find_data_files(directory):

    paths = []

    for (path, directories, filenames) in os.walk(directory):

        for filename in filenames:

            paths.append(os.path.join('..', path, filename))

    return paths

extra_files = find_data_files('pyspi/data')

setup(

    name="pyspi",
    packages=[
        'pyspi',
        'pyspi/io',
        'pyspi/utils',
        'pyspi/io'

    ],
    version='v1.0a',
    license='BSD',
    description='A python interface for INTEGRAL SPI',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',
    #   url = 'https://github.com/grburgess/pychangcooper',
 #   download_url='https://github.com/grburgess/pychangcooper/archive/1.1.2.tar.gz',

    package_data={'': extra_files, },
    include_package_data=True,

    install_requires=[
        'numpy',
        'matplotlib',
        'h5py',
        'pandas',
        'ipython',
        'astropy',
        'scipy',
        
    ],
)
