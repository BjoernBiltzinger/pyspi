import h5py
import numpy as np
import astropy.io.fits as fits
from glob import glob
import re
import os

def sort_human(l):
    """
    sort a list with indices and letters the way a human would
    :param l: a list of string
    """
    convert = lambda text: float(text) if text.isdigit() else text
    alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key) ]
    l.sort(key=alphanum)
    return l

def create_spi_irf_file(irf_database, file_name):
        
    irf_files = sort_human(glob(os.path.join(irf_database,
                                             'spi_irf_rsp*.fits')))[-51:]


    energies = np.zeros(len(irf_files),dtype=np.float64)

    masks = []

    for i, irf_file in enumerate(irf_files):
        with fits.open(irf_file) as f:

            irf_ext = f['SPI.-IRF.-RSP']

            irf_header = irf_ext.header

            # we only need to grab this once because it is always the same

            if i == 0:

                irf_crpix2 =  irf_header['CRPIX2']
                irf_crpix3 =   irf_header['CRPIX3']
                irf_crval2 =   irf_header['CRVAL2']
                irf_crval3 =   irf_header['CRVAL3']
                irf_cdelt2 =   irf_header['CDELT2']
                irf_cdelt3 =   irf_header['CDELT3']
                irf_reg =   irf_header['REGION']
                ndete = irf_header['NAXIS1']
                nx = irf_header['NAXIS2']
                ny = irf_header['NAXIS3']

            energies[i] = irf_header['ENERGY']

            # currently ignore the other two matrices

            masks.append(irf_ext.data.T[...,0])


    # now lots make one big matrix
    tmp = [len(masks)]
    tmp.extend(masks[0].shape)

    mask_matrix = np.zeros(tuple(tmp))

    for i,mask in enumerate(masks):

        mask_matrix[i,...] = mask

    f = h5py.File(file_name, "w")
    
    irf_dataset = f.create_dataset("irfs", 
                                   mask_matrix.shape,
                                   dtype=mask_matrix.dtype,
                                   compression="gzip")
    irf_dataset[...] = mask_matrix
    
    irf_dataset.attrs['irf_crpix2'] = irf_crpix2
    irf_dataset.attrs['irf_crpix3'] = irf_crpix3
    
    irf_dataset.attrs['irf_crval2'] = irf_crval2
    irf_dataset.attrs['irf_crval3'] = irf_crval3
    
    irf_dataset.attrs['irf_cdelt2'] = irf_cdelt2
    irf_dataset.attrs['irf_cdelt3'] = irf_cdelt3
    
    irf_dataset.attrs['irf_reg'] = irf_reg
    irf_dataset.attrs['ndete'] = ndete
    irf_dataset.attrs['nx'] = nx
    irf_dataset.attrs['ny'] = ny
    
    
    energies_dataset = f.create_dataset("energies", 
                                        energies.shape,
                                        dtype=energies.dtype,
                                        compression="gzip")
    
    energies_dataset[...] = energies
    
    
    f.close()
    
