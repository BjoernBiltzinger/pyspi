.. pySPI documentation master file, created by
   sphinx-quickstart on Sun Feb  4 11:24:43 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PySpi's documentation!
====================================
.. image:: media/pypsi_logo2.png

PySPI is pure python interface to analyze Gamma-Ray Burst (GRB) data from the spectrometer (SPI) onboard the International Gamma-Ray Astrophysics Laboratory (INTEGRAL). The INTEGRAL satellite is a gamma-ray observatory hosting four instruments that operate in the energy range between 3 keV and 10 MeV. It was launched in 2002 and is still working today. The main goals of PySPI are to provide an easy to install and develop analysis software for SPI, which includes improvements on the statistical analysis of GRB data. At the moment PySPI is designed for transient sources, like Gamma Ray Bursts (GRBs). In the future we plan to add support for other types of sources, such as persistent point sources as well as extended emission.

Comparison to OSA
------------------------------------

The main analysis tool to analyze SPI data up to now is the “Off-line Scientific Analysis” (OSA) `Chernyakova et al., 2020 <https://www.isdc.unige.ch/integral/download/osa/doc/11.1/osa_um_intro/man.html>`__), which is maintained by the INTEGRAL Science Data Centre (ISDC). While it is comprehensive in its capabilities for manipulating data obtained from all instrument on-board INTEGRAL, it exists as an IDL interface to a variety of low-level C++ libraries and is very difficult to install on modern computers. While there are containerized versions of OSA now available, the modern workflow of simply installing the software from a package manager and running on a local workstation is not possible and often students rely on a centralized installation which must be maintained by a seasoned expert. Moreover, adding more sophisticated and/or correct data analysis methods to the software requires an expertise that is not immediately accessible to junior researchers or non-experts in the installation of OSA. Also due to the increased computational power that is available today compared to that of 20 years ago, many of the analysis methods can be improved. PySPI addresses both these problems: It is providing an easy to install software, that can be developed further by everyone who wants to contribute. It also allows Bayesian fits of the data with true forward folding of the physical spectra into the data space via the response. This improves the sensitivity and the scientific output of GRB analyses with INTEGRAL/SPI.

Multi Mission Analysis
------------------------------------

PySPI provides a plugin for `3ML <https://threeml.readthedocs.io/en/stable>`__. This makes multi missions analysis with other instruments possible. Also all the spectral models from `astromodels <https://astromodels.readthedocs.io/en/latest/>`__ are available for the fits. Check out these two software packages for more information.


.. toctree::
   :maxdepth: 5
   :hidden:

   notebooks/installation.ipynb
   notebooks/time_series.ipynb
   notebooks/response.ipynb
   notebooks/psd.ipynb
   notebooks/active_detectors.ipynb
   notebooks/access_data.ipynb
   notebooks/contributing.ipynb
   api/API

.. nbgallery::
   :caption: Features and examples:

   notebooks/grb_analysis.ipynb
   notebooks/psd_eff.ipynb
