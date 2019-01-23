#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:41:19 2019

@author: tsiegert
"""

import sys
sys.executable

from spi_pointing import *
import pyspi
import matplotlib.pyplot as plt

# aux variables
# detector array
dets = np.arange(0,19,1)

# source position
src_ra, src_dec = 197.075, 58.9803

# energy bin of interest
ebounds = [40,40.5]

# for test use pointing index 4
pdx = 4

# pointing definition
pointing = SPIPointing('data/sc_orbit_param.fits.gz')
#pointing.sc_points[4]['scx_ra']
#pointing._sc_matrix.shape


# init response
spi = SPIResponse()

# calculate zenith and azimuth for all observation pointings
zenazi = pointing._calc_zenazi(src_ra,src_dec)
# choose the index
za = zenazi[pdx]

# calculate effective area for chosen energy bin
binned_arr = spi.get_binned_effective_area(np.deg2rad(za[1]),np.deg2rad(za[0]),ebounds)
# reform to 19-element array
binned_arr = np.reshape(binned_arr[0:19],19)
binned_arr_n = binned_arr/np.mean(binned_arr)

#print(binned_arr)

# spimodfit output
smf_conv = fits.open('data/convsky_model_SPI_GRB110903A_gamma2.fits')
smf_out = smf_conv['SPI.-BMOD-DSP'].data['COUNTS']
smf_out_n = smf_out/np.mean(smf_out)


plt.subplots(figsize=[12,5])
plt.subplot(1, 3, 1)
plt.step(dets,binned_arr,where='mid',label='pyspi output')
plt.step(dets,smf_out,where='mid',label='spimodfit output')
plt.xlim(-0.5,18.5)
plt.ylim(-0.5,9)
plt.xlabel('Detector ID')
plt.ylabel('Effective Area per Det. [cm$^2$]')
plt.legend()

plt.subplot(1,3,2)
plt.step(dets,binned_arr_n,where='mid',label='pyspi output')
plt.step(dets,smf_out_n,where='mid',label='spimodfit output')
plt.xlim(-0.5,18.5)
plt.ylim(-0.1,9)
plt.xlabel('Detector ID')
plt.ylabel('Effective Area per Det. normalised')
plt.legend()

plt.subplot(1,3,3)
plt.step(dets,(binned_arr_n-smf_out_n)/smf_out_n,where='mid',label='difference')
plt.plot([-1,20],[0,0],'k:')
plt.xlim(-0.5,18.5)
plt.ylim(-0.01,0.01)
plt.xlabel('Detector ID')
plt.ylabel('Relative difference (pyspi-smf)/smf')
plt.legend()
plt.tight_layout()