#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:41:19 2019

@author: tsiegert
"""

import numpy as np
from astropy.io import fits

import sys
sys.executable
import time

start = time.time()
from spi_pointing import *
end = time.time()
print("import spi_pointing: "+np.str(end-start))

start = time.time()
from spi_response import *
end = time.time()
print("import spi_response: "+np.str(end-start))

start = time.time()
import pyspi
end = time.time()
print("import pyspi: "+np.str(end-start))

import matplotlib.pyplot as plt


# aux variables
# detector array
dets = np.arange(0,19,1)

# source position
src_ra, src_dec = 197.075, 58.9803

# energy bin of interest
ebounds = [40,40.5]

# times
tstart = 4263.11378685185
tstop  = 4263.11413407407
telapse = tstop-tstart
telapse *= 86400

# for test use pointing index 4
pdx = 4

# pointing definition
pointing = SPIPointing('data/sc_orbit_param.fits.gz')
#pointing.sc_points[4]['scx_ra']
#pointing._sc_matrix.shape
data_file = fits.open('data/spi_oper.fits.gz')

response = SPIResponse()

times_sgl     = data_file['SPI.-OSGL-ALL'].data['TIME']
energies_sgl  = data_file['SPI.-OSGL-ALL'].data['ENERGY']
detectors_sgl = data_file['SPI.-OSGL-ALL'].data['DETE']
    
times_psd     = data_file['SPI.-OPSD-ALL'].data['TIME']
energies_psd  = data_file['SPI.-OPSD-ALL'].data['ENERGY']
detectors_psd = data_file['SPI.-OPSD-ALL'].data['DETE']

mask_sgl_t = (times_sgl >= tstart) & (times_sgl <= tstop)
mask_psd_t = (times_psd >= tstart) & (times_psd <= tstop)
    
mask_sgl_e = (energies_sgl >= 20) & (energies_sgl <= 8000)
mask_psd_e = (energies_psd >= 20) & (energies_psd <= 8000)
    
mask_sgl = mask_sgl_t & mask_sgl_e
mask_psd = mask_psd_t & mask_psd_e
    
mask_sgl_bg = ~mask_sgl_t & mask_sgl_e
mask_psd_bg = ~mask_psd_t & mask_psd_e

# temporary background definition
bg_sgl = energies_sgl[mask_sgl_bg]
bg_psd = energies_psd[mask_psd_bg]
bg     = np.append(bg_sgl,bg_psd)
    
bg_times = np.append(times_sgl[mask_sgl_bg],times_psd[mask_psd_bg])
T0_bg    = np.min(bg_times)
T1_bg    = np.max(bg_times)
DT_bg    = (T1_bg-T0_bg)*86400 - telapse

# source photons definition
src_sgl = energies_sgl[mask_sgl]
src_psd = energies_psd[mask_psd]
src     = np.append(src_sgl,src_psd)
det = np.append(detectors_sgl[mask_sgl], detectors_psd[mask_psd])
det_bg = np.append(detectors_sgl[mask_sgl_bg], detectors_psd[mask_psd_bg])
    
src_times = np.append(times_sgl[mask_sgl],times_psd[mask_psd])
T0_src    = np.min(src_times)
T1_src    = np.max(src_times)
DT_src    = (T1_src-T0_src)*86400


# calculate zenith and azimuth for all observation pointings
zenazi = pointing._calc_zenazi(src_ra,src_dec)
# choose the index
zen, azi = np.deg2rad(zenazi[pdx])

start = time.time()
aeff = response.get_effective_area(azi, zen, bg, det_bg)
end = time.time()
print("import pyspi: "+np.str(end-start))