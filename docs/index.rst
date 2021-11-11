.. pySPI documentation master file, created by
   sphinx-quickstart on Sun Feb  4 11:24:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySpi's documentation!
====================================
.. image:: media/pypsi_logo2.png

PySpi is pure python interface to analyze Integral/SPI data. At the moment it allows to fit GRB data. We plan to include non transient sources soon.

It provides a plugin for `3ML <https://threeml.readthedocs.io/en/stable>`__ and therefore all the spectral models from `astromodels <https://astromodels.readthedocs.io/en/latest/>`__ are available for the fits. Check out these two software packages for more information.

If you encounter any problems or bugs, please open an issue on `GitHub <https://github.com/BjoernBiltzinger/pyspi>`__ .

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
   notebooks/psd_eff.ipynb
