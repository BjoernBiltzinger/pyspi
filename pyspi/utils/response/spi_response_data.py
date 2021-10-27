import numpy as np
import h5py
from dataclasses import dataclass

from pyspi.io.package_data import get_path_of_data_file
from pyspi.utils.rmf_base import load_rmf_non_ph_1, load_rmf_non_ph_2

@dataclass
class ResponseData:
    """
    Base Datacĺass to hold the IRF data
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
        :return: all the infomation we need as a list
        """
        assert version in [0, 1, 2, 3, 4],\
            f"Version must be in [0, 1, 2, 3, 4] but is {version}"

        irf_file =\
            get_path_of_data_file(f"spi_three_irfs_database_{version}.hdf5")

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
            irf_ybin, irf_nx, irf_ny, irf_data, ebounds_rmf_2_base, rmf_2_base, \
            ebounds_rmf_3_base, rmf_3_base

@dataclass
class ResponseDataPhotopeak(ResponseData):
    """
    Datacĺass to hold the IRF data if we only need the photopeak irf
    """
    irfs_photopeak: np.array

    @classmethod
    def from_version(cls, version):
        """
        Construct the dataclass object
        :param version: Which IRF version?
        :return: ResponseIRFReadPhotopeak object
        """
        data = super().get_data(ResponseData, version)

        irfs_photopeak = data[0][:, :, :, :, 0]

        return cls(*data[1:],
                   irfs_photopeak)


@dataclass
class ResponseDataRMF(ResponseData):
    """
    Datacĺass to hold the IRF data if we only need all three irfs
    """
    irfs_photopeak: np.array
    irfs_nonphoto_1: np.array
    irfs_nonphoto_2: np.array

    @classmethod
    def from_version(cls, version):
        """
        Construct the dataclass object
        :param version: Which IRF version?
        :return: ResponseIRFReadPhotopeak object
        """
        data = super().get_data(ResponseData, version)

        irfs_photopeak = data[0][:, :, :, :, 0]
        irfs_nonphoto_1 = data[0][:, :, :, :, 1]
        irfs_nonphoto_2 = data[0][:, :, :, :, 2]

        return cls(*data[1:],
                   irfs_photopeak,
                   irfs_nonphoto_1,
                   irfs_nonphoto_2)
