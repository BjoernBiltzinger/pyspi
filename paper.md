---
title: 'PySPI: A python analysis framework for INTEGRAL/SPI'
tags:
  - Python
  - Astronomy
  - Gamma-Ray Bursts

authors:
  - name: Björn Biltzinger
    orcid: 0000-0003-3345-9515
    affiliation: "1, 2"
  - name: J. Michael Burgess
    orcid: 0000-0003-3345-9515
    affiliation: "1"
  - name: Thomas Siegert
    orcid: 0000-0002-1153-2139
    affiliation: "1"
  - name: Jochen Greiner
    orcid: 0000-0002-1153-2139
    affiliation: "1"
bibliography: paper.bib
affiliations:
 - name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse, 85748 Garching, Germany
   index: 1
 - name: Technical University of Munich, Boltzmannstrasse 2, 85748 Garching, Germany
   index: 2
date: "27 October 2021"
---

# Summary

The INTEGRAL satellite was launched in 2002 and is still working. PySPI is a newly developed pure python analysis framework to analysis Gamma-Ray Burst (GRB) data from the instrument SPectrometer on INTEGRAL (SPI), which is onboard of INTEGRAL. Its main goals are to provide a easy to install and develop analysis software, which includes improvements on the statistical analysis of GRB data. 

In the future we plan to add support for other sources than GRBs, like constant sources (point and extended). 

# Statement of need

The main analysis tools to analyze the data up to now are the "Off-line Scientific Analysis" (OSA) [@osa], which are maintained by the INTEGRAL Science Data Centre (ISDC). While they are comprehensive in their capabilities for manipulating data obtained from all instrument on-board INTEGRAL, they exists as an IDL interface to a variety of low-level C++ libraries. These libraries are large and are difficult to uniformly distribute and install on modern computer workstations. While there are containerized versions of OSA now available, the modern workflow of simply installing a the software from a package manager and running on a local workstation is not possible and often students rely on a centralized installations which must be maintained by a seasoned expert. Moreover, adding more sophisticated and/or correct data analysis methods to the software requires an expertise that is not immediately accessible to junior researchers or non-experts in the installation of OSA.

Also due to the increased computational power that is available today compared to 20 years ago, many of the analysis methods can be improved. PySPI addresses both these problems. It is providing a simply to install software, that can be easily developed further by everyone who wants to participate. And it also allows Bayesian fits of the data with true forward folding of the physical spectra into the data space via the response, which improves the scientific output of GRB analysis with INTEGRAL/SPI. 

# Procedure

To analyze GRB data PySPI needs a few inputs, like the time of the GRB and the energy bins that should be used in the analysis. With these information, it automatically downloads all the data files it needs for this specific analysis. These are then used to construct a response and a time series for the observation that contains the GRB time. With the help of these time series one can select active time intervals for the source that should be used in the fits and time intervals before and after the GRB signal for background estimation. After this has been done a plugin for `3ML` [@3mlpaper @3ML] can be constructed. This allows for all the benefits, the 3ML framework offers, like the modeling framework `astromodels`[@astromodels], joined fits with other instruments, many different bayesian sampler and much more. 

# Acknowledgments



# References

