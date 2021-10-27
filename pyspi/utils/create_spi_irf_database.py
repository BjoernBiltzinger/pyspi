import h5py
import numpy as np
import astropy.io.fits as fits
from glob import glob
import re
import os


def create_spi_irf_file(irf_grp_fits_file):
    """
    Create the IRF hdf5 file pyspi needs to run. To use this function you need
    access to the mpe afs. But normal users never need to run this function,
    because the result should be in the data folder.
    """

    version_number = int(irf_grp_fits_file.split(".")[-2][-2:])-20
    # for some reason version number 5 was skipped
    if version_number > 4:
        version_number -= 1

    # in pyspi we start counting at 0
    version_number -= 1

    base = os.path.join(*irf_grp_fits_file.split("/")[:-1])
    base = f"/{base}"

    with fits.open(irf_grp_fits_file) as f:
        irf_files = f[2].data["MEMBER_LOCATION"]

    energies = np.zeros(len(irf_files),dtype=np.float64)

    masks = []

    for i, irf_file in enumerate(irf_files):
        print(irf_file)
        with fits.open(os.path.join(base, irf_file)) as f:

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
                irf_xmin = np.deg2rad(irf_crval2 - np.floor(irf_crpix2-0.5) * irf_cdelt2)
                irf_ymin  = np.deg2rad(irf_crval3 - np.floor(irf_crpix3-0.5) * irf_cdelt3)
                irf_xbin  = np.deg2rad(irf_cdelt2)
                irf_ybin  =  np.deg2rad(irf_cdelt3)

            energies[i] = irf_header['ENERGY']

            # Get all three IRF values, for photo peak, non-photo peak that first interact in det and
            # non-photo peak that first interact in the dead material

            masks.append(irf_ext.data.T)

    # now lots make one big matrix
    tmp = [len(masks)]
    tmp.extend(masks[0].shape)

    mask_matrix = np.zeros(tuple(tmp))

    for i, mask in enumerate(masks):

        mask_matrix[i,...] = mask

    f = h5py.File(f"spi_three_irfs_database_{version_number}.hdf5", "w")

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

    irf_dataset.attrs['irf_xmin'] = irf_xmin
    irf_dataset.attrs['irf_ymin'] = irf_ymin
    irf_dataset.attrs['irf_xbin'] = irf_xbin
    irf_dataset.attrs['irf_ybin'] = irf_ybin

    energies_dataset = f.create_dataset("energies",
                                        energies.shape,
                                        dtype=energies.dtype,
                                        compression="gzip")

    energies_dataset[...] = energies

    f.close()
