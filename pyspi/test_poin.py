#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 13:41:19 2019

@author: tsiegert
"""

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


# definition of burst data and background times / events
if (1):
    start = time.time()
    times_sgl     = data_file['SPI.-OSGL-ALL'].data['TIME']
    energies_sgl  = data_file['SPI.-OSGL-ALL'].data['ENERGY']
    detectors_sgl = data_file['SPI.-OSGL-ALL'].data['DETE']
    
    times_psd     = data_file['SPI.-OPSD-ALL'].data['TIME']
    energies_psd  = data_file['SPI.-OPSD-ALL'].data['ENERGY']
    detectors_psd = data_file['SPI.-OPSD-ALL'].data['DETE']
    
    tdx_sgl = np.where((times_sgl >= tstart) & (times_sgl <= tstop))
    tdx_psd = np.where((times_psd >= tstart) & (times_psd <= tstop))
    
    edx_sgl = np.where((energies_sgl >= 20) & (energies_sgl <= 8000))
    edx_psd = np.where((energies_psd >= 20) & (energies_psd <= 8000))
    
    tedx_sgl = np.where((times_sgl >= tstart) & (times_sgl <= tstop) & \
                        (energies_sgl >= 20) & (energies_sgl <= 8000))
    tedx_psd = np.where((times_psd >= tstart) & (times_psd <= tstop) & \
                        (energies_psd >= 20) & (energies_psd <= 8000))
    
    tedx_sgl_bg = np.where(((times_sgl < tstart) | (times_sgl > tstop)) & \
                           (energies_sgl >= 20) & (energies_sgl <= 8000))
    tedx_psd_bg = np.where(((times_psd < tstart) | (times_psd > tstop)) & \
                           (energies_psd >= 20) & (energies_psd <= 8000))
    
    # temporary background definition
    bg_sgl = energies_sgl[tedx_sgl_bg]
    bg_psd = energies_psd[tedx_psd_bg]
    bg     = np.append(bg_sgl,bg_psd)
    
    bg_times = np.append(times_sgl[tedx_sgl_bg],times_psd[tedx_psd_bg])
    T0_bg    = np.min(bg_times)
    T1_bg    = np.max(bg_times)
    DT_bg    = (T1_bg-T0_bg)*86400 - telapse
    
    # per detector
    
    teddx_sgl_bg = {}
    for i in range(19):
        teddx_sgl_bg['D'+str(i).zfill(2)] = np.where(((times_sgl < tstart) | (times_sgl > tstop)) & \
                             (energies_sgl >= 20) & (energies_sgl <= 8000) & \
                             (detectors_sgl == i))
    
    teddx_psd_bg = {}
    for i in range(19):
        teddx_psd_bg['D'+str(i).zfill(2)] = np.where(((times_psd < tstart) | (times_psd > tstop)) & \
                             (energies_psd >= 20) & (energies_psd <= 8000) & \
                             (detectors_psd == i))
    
    # source photons definition
    src_sgl = energies_sgl[tedx_sgl]
    src_psd = energies_psd[tedx_psd]
    src     = np.append(src_sgl,src_psd)
    
    src_times = np.append(times_sgl[tedx_sgl],times_psd[tedx_psd])
    T0_src    = np.min(src_times)
    T1_src    = np.max(src_times)
    DT_src    = (T1_src-T0_src)*86400
    
    src_detectors = np.append(detectors_sgl[tedx_sgl],detectors_psd[tedx_psd])
    src_energies = src
    
    end = time.time()
    print("source and background array definitions: "+np.str(end-start))

    # init response
    start = time.time()
    spi = SPIResponse()
    end = time.time()
    print("init SPI response: "+np.str(end-start))

# calculate zenith and azimuth for all observation pointings
zenazi = pointing._calc_zenazi(src_ra,src_dec)
# choose the index
za = np.deg2rad(zenazi[pdx])
#za = zenazi

# calculate effective area for chosen energy bin
binned_arr = spi.get_binned_effective_area(za[1],za[0],ebounds)
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

start = time.time()
if (1):
    bins = np.arange(20,8001,1)
    h_bg = np.histogram(bg,bins=bins)
    h_bg_sgl = np.histogram(bg_sgl,bins=bins)
    h_bg_psd = np.histogram(bg_psd,bins=bins)
    h_src_sgl = np.histogram(src_sgl,bins=bins)
    h_src_psd = np.histogram(src_psd,bins=bins)
    h_src = np.histogram(src,bins=bins)
    loc_bg  = h_bg[1][0:-1] + 0.5
    loc_src = h_src[1][0:-1] + 0.5
    loc     = loc_bg
end = time.time()
print("Fine histograms: "+np.str(end-start))

hist_bg     = h_bg[0]
hist_src    = h_src[0]
hist_bg_sgl = h_bg_sgl[0]
hist_bg_psd = h_bg_psd[0]
hist_src_sgl = h_src_sgl[0]
hist_src_psd = h_src_psd[0]


plaw = 0.15*(src_energies/100.)**(-0.4)*(np.exp(-src_energies/500.))
aeff_src = spi.get_effective_area(za[1],za[0],src_energies,src_detectors)

plt.figure(figsize=[10.24,7.68])
plt.step(loc_bg,hist_src/DT_src-hist_bg/DT_bg)
#plt.loglog(bin_ndarray((np.sort(src_energies))[0:24000],new_shape=(100,),operation='mean'), \
#           bin_ndarray(((plaw*aeff_src)[np.argsort(src_energies)])[0:24000],new_shape=(100,),operation='mean'),'-ro')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('cnt/s')
plt.xlim(15,9000)
plt.ylim(1e-3,1e1)
