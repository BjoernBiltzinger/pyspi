---
title: 'PySPI: A python analysis framework for INTEGRAL/SPI'
tags:
  - Python
  - Astronomy
  - Gamma-Ray Bursts
  - INTEGRAL/SPI
authors:
  - name: Björn Biltzinger
    orcid: 0000-0003-3381-0985
    affiliation: "1, 2"
  - name: J. Michael Burgess
    orcid: 0000-0003-3345-9515
    affiliation: "1"
  - name: Thomas Siegert
    orcid: 0000-0002-0552-3535
    affiliation: "1"
bibliography: paper.bib
affiliations:
 - name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse 1, 85748 Garching, Germany
   index: 1
 - name: Technical University of Munich, Boltzmannstrasse 2, 85748 Garching, Germany
   index: 2
date: "27 October 2021"
---

# Summary

PySPI is a newly developed pure python analysis framework for 
Gamma-Ray Burst (GRB) data from the spectrometer (SPI) onboard the 
International Gamma-Ray Astrophysics Laboratory (INTEGRAL). The INTEGRAL 
satellite is a gamma-ray observatory hosting four instruments that 
operate in the energy range between 3 keV and 10 MeV. It was launched in 
2002 and is still working today. The main goals of PySPI are to provide 
an easy to install and develop analysis software for SPI, which includes 
improvements on the statistical analysis of GRB data.

At the moment PySPI is designed for transient sources. One interesting example of 
transient sources are GRBs, which are extremely bright but short flashes 
of Gamma-Rays, with a typical duration between a few ms and a few hundred seconds. They are
believed to be produced by the collapse of massive stars and
mergers of compact objects, like for example neutron stars. In the future we plan to add support for other types of sources than transients, such as persistent point sources as well as extended emission.


# Statement of need

The main analysis tool to analyze SPI data up to now is the
"Off-line Scientific Analysis" (OSA) [@osa], which is maintained by
the INTEGRAL Science Data Centre (ISDC). While it is comprehensive
in its capabilities for manipulating data obtained from all
instrument on-board INTEGRAL, it exists as an IDL interface to a
variety of low-level C++ libraries and is very difficult to install on modern computers. 
While there are containerized 
versions of OSA now available, the modern workflow of simply installing 
the software from a package manager and running on a local workstation is 
not possible and often students rely on a centralized installation which 
must be maintained by a seasoned expert. Moreover, adding more sophisticated 
and/or correct data analysis methods to the software requires an expertise 
that is not immediately accessible to junior researchers or non-experts in 
the installation of OSA. Also due to the
increased computational power that is available today compared to that
of 20 years ago, many of the analysis methods can be improved.

PySPI addresses both these problems: It is providing an easy to install 
software, that can be developed further by everyone who wants to participate. 
It also allows Bayesian fits of the data with true forward folding of the physical spectra 
into the data space via the response. This improves the sensitivity 
and the scientific output of GRB analyses with INTEGRAL/SPI. 


# SPectrometer on INTEGRAL (SPI)

SPI is a coded mask instrument covering the energy range between 20 keV and 8 MeV. It consists of a detector plane with 19 Germanium detectors and a mask plane 1.7 meters above the detectors with 3 cm thick tungsten elements. The mask produces a shadow pattern on the detectors depending on the source position. This information can be used to construct an image from an observation. Also SPI has an excellent energy resolution of 2.5 keV at 1.3 MeV, which makes SPI an ideal instrument to analyze fine spectral features, such as lines from radioactive decays [@spi]. 


# Procedure

To analyze GRB data, PySPI accepts inputs such as the time of the GRB
and the spectral energy bins that will be used in an analysis. With this
information, it automatically downloads all the data files required
for a specific analysis and constructs a response 
as well as a time series for the observation that contains the GRB time. The time series 
can be used to select active time intervals for the source, and time intervals before 
and after the GRB signal for background estimation. After this has been done, a plugin 
for `3ML` [@3mlpaper;@3ML] can be constructed. This allows for all the benefits the 3ML 
framework offers like the modeling framework `astromodels` [@astromodels], joint 
fits with other instruments, many different Bayesian samplers and much more. 
In the [documentation](https://pyspi.readthedocs.io/en/latest/) there is an
[example](https://pyspi.readthedocs.io/en/latest/notebooks/grb_analysis/)
for this workflow procedure.


# Acknowledgments

B. Biltzinger acknowledges financial support from the `German Aerospaces Center (Deutsches Zentrum für Luft- und Raumfahrt, DLR)` under FKZ 50 0R 1913. 


# References

