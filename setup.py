#from setuptools import setup
from distutils.core import setup
setup(

    name="pyspi",
    packages=[
        'pyspi',

    ],
    version='v1.0a',
    license='BSD',
    description='A generic chang and cooper solver for fokker-planck equations',
    author='J. Michael Burgess',
    author_email='jmichaelburgess@gmail.com',
 #   url = 'https://github.com/grburgess/pychangcooper',
 #   download_url='https://github.com/grburgess/pychangcooper/archive/1.1.2.tar.gz',
    requires=[
        'numpy',
        'matplotlib'
    ],
)
