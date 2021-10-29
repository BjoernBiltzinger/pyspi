import numpy as np
import h5py
from dataclasses import dataclass
import os
import astropy.io.fits as fits

from pyspi.io.package_data import get_path_of_internal_data_dir


def load_rmf_non_ph_1():
    """
    Load the RMF for the non-photopeak events that first interact in the det

    :returns: ebounds of RMF and rmf matrix for the non-photopeak events that
        first interact in the det
    """

    with fits.open(os.path.join(
            get_path_of_internal_data_dir(),
            'spi_rmf2_rsp_0002.fits')
                   ) as rmf_file:
        rmf_comp = rmf_file['SPI.-RMF2-RSP'].data['MATRIX']
        emax = rmf_file['SPI.-RMF2-RSP'].data['ENERG_HI']
        emin = rmf_file['SPI.-RMF2-RSP'].data['ENERG_LO']

    ebounds = np.append(emin, emax[-1])

    # The RMFs are stored in a weird way, we have to expand this to a
    # real square matrix
    rmf = np.zeros((len(rmf_comp), len(rmf_comp)))
    for i, row in enumerate(rmf_comp):
        length = len(row.flatten())
        rmf[i, :length] = row.flatten()

    return ebounds, rmf


def load_rmf_non_ph_2():
    """
    Load the RMF for the non-photopeak events that first interact
    in the dead material

    :returns: ebounds of RMF and rmf matrix for the non-photopeak events that
        first interact in the dead material
    """

    with fits.open(os.path.join(
            get_path_of_internal_data_dir(),
            'spi_rmf3_rsp_0002.fits')
                   ) as rmf_file:
        rmf_comp = rmf_file['SPI.-RMF3-RSP'].data['MATRIX']
        emax = rmf_file['SPI.-RMF3-RSP'].data['ENERG_HI']
        emin = rmf_file['SPI.-RMF3-RSP'].data['ENERG_LO']

    ebounds = np.append(emin, emax[-1])

    # The RMFs are stored in a weird way, we have to expand this to a
    # real square matrix
    rmf = np.zeros((len(rmf_comp), len(rmf_comp)))
    for i, row in enumerate(rmf_comp):
        length = len(row.flatten())
        rmf[i, :length] = row.flatten()
    rmf[0] = np.zeros(len(rmf))
    return ebounds, rmf


@dataclass
class ResponseData:
    """
    Base Dataclass to hold the IRF data
    """

    energies_database: np.array
    irf_xmin: float
    irf_ymin: float
    irf_xbin: float
    irf_ybin: float
    irf_nx: int
    irf_ny: int
    n_dets: int
    ebounds_rmf_2_base: np.array
    rmf_2_base: np.array
    ebounds_rmf_3_base: np.array
    rmf_3_base: np.array

    def get_data(self, version):
        """
        Read in the data we need from the irf hdf5 file

        :param version: Version of irf file

        :returns: all the infomation we need as a list
        """
        assert version in [0, 1, 2, 3, 4],\
            f"Version must be in [0, 1, 2, 3, 4] but is {version}"

        irf_file = os.path.join(get_path_of_internal_data_dir(),
                                f"spi_three_irfs_database_{version}.hdf5")

        if version == 0:
            print('Using the irfs that are valid between Start'
                  ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')

        elif version == 1:
            print('Using the irfs that are valid between 03/07/06 06:00:00'
                  ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif version == 2:
            print('Using the irfs that are valid between 04/07/17 08:20:06'
                  ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif version == 3:
            print('Using the irfs that are valid between 09/02/19 09:59:57'
                  ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            print('Using the irfs that are valid between 10/05/27 12:45:00'
                  ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')
        energies_database = irf_database['energies'][()]
        irf_data = irf_database['irfs']
        irfs = irf_data[()]
        irf_xmin = irf_data.attrs['irf_xmin']
        irf_ymin = irf_data.attrs['irf_ymin']
        irf_xbin = irf_data.attrs['irf_xbin']
        irf_ybin = irf_data.attrs['irf_ybin']
        irf_nx = irf_data.attrs['nx']
        irf_ny = irf_data.attrs['ny']
        irf_database.close()

        ebounds_rmf_2_base, rmf_2_base = load_rmf_non_ph_1()
        ebounds_rmf_3_base, rmf_3_base = load_rmf_non_ph_2()

        return irfs, energies_database, irf_xmin, irf_ymin, irf_xbin, \
            irf_ybin, irf_nx, irf_ny, irf_data, ebounds_rmf_2_base, \
            rmf_2_base, ebounds_rmf_3_base, rmf_3_base


@dataclass
class ResponseDataPhotopeak(ResponseData):
    """
    Dataclass to hold the IRF data if we only need the photopeak irf
    """
    irfs_photopeak: np.array

    @classmethod
    def from_version(cls, version):
        """
        Construct the dataclass object

        :param version: Which IRF version?

        :returns: ResponseDataPhotopeak object
        """
        data = super().get_data(ResponseData, version)

        irfs_photopeak = data[0][:, :, :, :, 0]

        return cls(*data[1:],
                   irfs_photopeak)


@dataclass
class ResponseDataRMF(ResponseData):
    """
    Dataclass to hold the IRF data if we only need all three irfs
    """
    irfs_photopeak: np.array
    irfs_nonphoto_1: np.array
    irfs_nonphoto_2: np.array

    @classmethod
    def from_version(cls, version):
        """
        Construct the dataclass object

        :param version: Which IRF version?

        :returns: ResponseDataPhotopeak object
        """
        data = super().get_data(ResponseData, version)

        irfs_photopeak = data[0][:, :, :, :, 0]
        irfs_nonphoto_1 = data[0][:, :, :, :, 1]
        irfs_nonphoto_2 = data[0][:, :, :, :, 2]

        return cls(*data[1:],
                   irfs_photopeak,
                   irfs_nonphoto_1,
                   irfs_nonphoto_2)
