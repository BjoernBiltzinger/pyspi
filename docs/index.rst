.. pySPI documentation master file, created by
   sphinx-quickstart on Sun Feb  4 11:24:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySpi's documentation!
====================================
.. image:: media/pypsi_logo2.png

PySpi is pure python interface to analyze Integral/SPI data. At the moment it allows to fit transient and constant point source and we want to include extended sources soon.

It provides plugins for `3ML <https://threeml.readthedocs.io/en/stable>`__ and therefore all the spectral models from `astromodels <https://astromodels.readthedocs.io/en/latest/>`__ are available for the fits. Check out these two software packages for more information.

This is still not a stable version. Bugs will occur sometimes, please open an issue on github if you find one.

.. toctree::
   :maxdepth: 5
   :hidden:

   notebooks/installation.ipynb
   notebooks/time_series.ipynb
   notebooks/response.ipynb
   notebooks/psd.ipynb
   notebooks/active_detectors.ipynb
   api/API

.. nbgallery::
   :caption: Features and examples:

   notebooks/grb_analysis.ipynb
