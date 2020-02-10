import numpy as np
import astropy.io.fits as fits

from pyspi.io.package_data import get_path_of_data_file

def load_rmf_non_ph_1():
    """
    Load the RMF for the non-photopeak events that first interact in the det
    """
    
    with fits.open(get_path_of_data_file('spi_rmf2_rsp_0002.fits')) as f:
        rmf_comp = f['SPI.-RMF2-RSP'].data['MATRIX']
        emax = f['SPI.-RMF2-RSP'].data['ENERG_HI']
        emin = f['SPI.-RMF2-RSP'].data['ENERG_LO']

    ebounds = np.append(emin, emax[-1])
    rmf = np.zeros((len(rmf_comp), len(rmf_comp)))
    for i in range(len(rmf_comp)):
        length = len(rmf_comp[i].flatten())
        rmf[i, :length] = rmf_comp[i].flatten()

    return ebounds, rmf
    
def load_rmf_non_ph_2():
    """
    Load the RMF for the non-photopeak events that first interact in the dead material
    """

    with fits.open(get_path_of_data_file('spi_rmf3_rsp_0002.fits')) as f:
        rmf_comp = f['SPI.-RMF3-RSP'].data['MATRIX']
        emax = f['SPI.-RMF3-RSP'].data['ENERG_HI']
        emin = f['SPI.-RMF3-RSP'].data['ENERG_LO']

    ebounds = np.append(emin, emax[-1])
    rmf = np.zeros((len(rmf_comp), len(rmf_comp)))
    for i in range(len(rmf_comp)):
        length = len(rmf_comp[i].flatten())
        rmf[i, :length] = rmf_comp[i].flatten()
    rmf[0] = np.zeros(len(rmf))
    return ebounds, rmf
