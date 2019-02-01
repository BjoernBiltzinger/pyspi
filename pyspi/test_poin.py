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

start = time.time()
from spi_background import *
end = time.time()
print("import pyspi: "+np.str(end-start))

import matplotlib.pyplot as plt


# aux variables
# detector array
dets = np.arange(0,19,1)
bins = np.arange(20,8001,1)

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
    
    tdx_sgl = ((times_sgl >= tstart) & (times_sgl <= tstop))
    tdx_psd = ((times_psd >= tstart) & (times_psd <= tstop))
    
    edx_sgl = ((energies_sgl >= 20) & (energies_sgl <= 8000))
    edx_psd = ((energies_psd >= 20) & (energies_psd <= 8000))
    
    tedx_sgl = ((times_sgl >= tstart) & (times_sgl <= tstop) & \
                        (energies_sgl >= 20) & (energies_sgl <= 8000))
    tedx_psd = ((times_psd >= tstart) & (times_psd <= tstop) & \
                        (energies_psd >= 20) & (energies_psd <= 8000))
    
    tedx_sgl_bg = (((times_sgl < tstart) | (times_sgl > tstop)) & \
                           (energies_sgl >= 20) & (energies_sgl <= 8000))
    tedx_psd_bg = (((times_psd < tstart) | (times_psd > tstop)) & \
                           (energies_psd >= 20) & (energies_psd <= 8000))
    
    end = time.time()
    print("source and background array definitions 1: "+np.str(end-start))
    
    # temporary background definition
    bg_sgl = energies_sgl[tedx_sgl_bg]
    bg_psd = energies_psd[tedx_psd_bg]
    bg     = np.append(bg_sgl,bg_psd)
    
    bg_times = np.append(times_sgl[tedx_sgl_bg],times_psd[tedx_psd_bg])
    T0_bg    = np.min(bg_times)
    T1_bg    = np.max(bg_times)
    DT_bg    = (T1_bg-T0_bg)*86400 - telapse
    
    end = time.time()
    print("source and background array definitions 2: "+np.str(end-start))
    
    # per detector
    if (0):
        teddx_sgl_bg = {}
        for i in range(19):
            teddx_sgl_bg['D'+str(i).zfill(2)] = (((times_sgl < tstart) | (times_sgl > tstop)) & \
                                 (energies_sgl >= 20) & (energies_sgl <= 8000) & \
                                 (detectors_sgl == i))
        
        teddx_psd_bg = {}
        for i in range(19):
            teddx_psd_bg['D'+str(i).zfill(2)] = (((times_psd < tstart) | (times_psd > tstop)) & \
                                 (energies_psd >= 20) & (energies_psd <= 8000) & \
                                 (detectors_psd == i))
        
        
        bg_grb_dets_sgl = np.zeros((7980,len(dets)))
        bg_grb_dets_psd = np.zeros((7980,len(dets)))
        for i in range(19):
            bg_grb_dets_sgl[:,i] = np.histogram(energies_sgl[teddx_sgl_bg['D'+str(i).zfill(2)]],bins=bins)[0]
            bg_grb_dets_psd[:,i] = np.histogram(energies_psd[teddx_psd_bg['D'+str(i).zfill(2)]],bins=bins)[0]
        
        work_dets = [0,3,4,6,7,8,9,10,11,12,13,14,15,16,18]
        bg_grb_dets_sgl[:,work_dets] += 1
        bg_grb_dets_psd[:,work_dets] += 1
        
        
        # create energy locations array + hdu
        emin = bins[0:-1]
        emax = bins[1:]
        loc = bins[0:-1]+0.5
        hdu_erg = fits.BinTableHDU.from_columns(
                [fits.Column(name='EMIN',format='E',array=emin,unit='keV'),
                 fits.Column(name='EMAX',format='E',array=emax,unit='keV'),
                 fits.Column(name='ECEN',format='E',array=loc,unit='keV')])
        hdu_erg.name = 'ENERGIES'
        # write SPI GRB background model template for epoch 5
        hdu = fits.BinTableHDU.from_columns(
                [fits.Column(name='BG_SGL', format='7980E', array=bg_grb_dets_sgl.T/DT_bg + 0.00030266, unit='1/s'),
                 fits.Column(name='BG_PSD', format='7980E', array=bg_grb_dets_psd.T/DT_bg + 0.00030266, unit='1/s')])
        hdu.name = 'SPI.-GRB-BG05'
        # create primary header because we apparenly must
        hdr = fits.Header()
        hdr['CREATED'] = 'Thomas Siegert'
        hdr['COMMENT'] = "This is a PYSPI GRB template file. Here, only Epoch 5 (i.e. 15 working detectors, 4 failures, starting at revolution 930,) is included."
        hdr['COMMENT'] = "More to be added."
        hdr['ENE_UNIT'] = "keV"
        hdr['BGM_UNIT'] = "1/s"
        primary_hdu = fits.PrimaryHDU(header=hdr)
        # combine HDUs and write
        hdul = fits.HDUList([primary_hdu,hdu_erg,hdu])
        hdul.writeto('spi_grb_background.fits')
        
    
    
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
    
    data_energies_sgl = energies_sgl[tedx_sgl]
    data_energies_psd = energies_psd[tedx_psd]
    data_energies = np.append(data_energies_sgl,data_energies_psd)
    data_detectors_sgl = detectors_sgl[tedx_sgl]
    data_detectors_psd = detectors_psd[tedx_psd]
    data_detectors = np.append(data_detectors_sgl,data_detectors_psd)
    data_evtstypes_sgl = np.zeros(len(np.where(tedx_sgl == True)[0]),dtype=np.int)
    data_evtstypes_psd = np.ones(len(np.where(tedx_psd == True)[0]),dtype=np.int)
    data_evtstypes = np.append(data_evtstypes_sgl,data_evtstypes_psd)
    
    
    # hier weiter mit detectors und event type
    
    
    end = time.time()
    print("source and background array definitions 3: "+np.str(end-start))

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


plaw = 0.10*(src_energies/100.)**(-1.0)#*(np.exp(-src_energies/500.))
aeff_src = spi.get_effective_area(za[1],za[0],src_energies,src_detectors)

plt.figure(figsize=[10.24,7.68])
plt.step(loc_bg,hist_src/DT_src-hist_bg/DT_bg)
plt.plot(bin_ndarray((np.sort(data_energies))[0:24000],new_shape=(100,),operation='mean'), \
         bin_ndarray(((plaw*aeff_src)[np.argsort(src_energies)])[0:24000],new_shape=(100,),operation='mean'),'-ro')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Energy [keV]')
plt.ylabel('cnt/s')
plt.xlim(15,9000)
plt.ylim(1e-3,1e1)
