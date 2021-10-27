---
title: 'PySPI: A python analysis framework for INTEGRAL/SPI'
tags:
  - Python
  - Astronomy
  - Gamma-Ray Bursts

authors:
  - name: Björn Biltzinger
    orcid: 0000-0003-3345-9515
    affiliation: "1", "2"
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

The INTEGRAL satellite was launched in 2002 and is still working. PySPI is newly developed pure python analysis framework to analysis Gamma-Ray Burst (GRB) data from the instrument SPectrometer on INTEGRAL (SPI), which is onboard of INTEGRAL. Its main goals are to provide a easy to install and develop analysis software, which includes improvements on the statistical analysis of GRB data. 

In the future we plan to add support for other sources than GRBs, like constant sources (point and extended). 

# Statement of need

The main analysis tools to analyze the data up to now are the "Off-line Scientific Analysis" (OSA) (cite OSA), which are maintained by the INTEGRAL Science Data Centre (ISDC). These tools are written in IDL and very hard to install on modern computer. Often the only solution to run these tools nowadays is to use a docker image, which makes further development by the community very difficult. Also due to the increased computational power that is available today compared to 20 years ago, many of the analysis methods can be improved. PySPI addresses both these problems. It providing a simply to install software, that can be easily developed further by everyone who wants to participate. And it also allows Bayesian fits of the data with true forward folding of the physical spectra into the data space via the response, which improves the scientific output of GRB analysis with INTEGRAL/SPI. 

# Procedure

To analyze GRB data PySPI needs a few inputs, like for example the time of the GRB and the energy bins that should be used in the analysis. With these information, it automatically downloads all the data files it needs for this specific analysis. These are then used to construct a response and a time series for the observation that contains the GRB time. With the help of these time series one can select active time intervals for the source that should be used in the fits and time intervals before and after the GRB signal for background estimation. After this has been done a plugin for 3ML (cite and href) can be constructed. This allows for all the benefits, the 3ML framework offers, like the modeling framework astromodels (cite and href), joined fits with other instruments, many different bayesian sampler and much more. 

# Acknowledgments

# References

