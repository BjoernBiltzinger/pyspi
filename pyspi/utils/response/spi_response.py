import os
from datetime import datetime

import h5py
import numpy as np
import scipy.interpolate as interpolate
import yaml
from astropy.time.core import Time, TimeDelta
from IPython.display import HTML
from numba import float64, njit
from threeML.io.file_utils import sanitize_filename

from interpolation import interp



from pyspi.config.config_builder import Config
from pyspi.io.get_files import get_files_afs, get_files_isdcarc
from pyspi.io.package_data import (get_path_of_data_file,
                                   get_path_of_external_data_dir)
from pyspi.utils.function_utils import construct_energy_bins, find_needed_ids
from pyspi.utils.response.spi_pointing import (SPIPointing,
                                               _construct_sc_matrix,
                                               _transform_icrs_to_spi)
#import scipy.integrate as integrate
from pyspi.utils.rmf_base import *
from pyspi.utils.rmf_base import load_rmf_non_ph_1, load_rmf_non_ph_2


@njit([float64[:](float64[:,::1], float64[:,::1])])
def trapz(y,x):
    """
    Fast trapz integration with numba
    :param x: x values
    :param y: y values
    :return: Trapz integrated
    """
    return np.trapz(y,x)

@njit#(float64[:](float64[:], float64[:], float64[:]))
def log_interp1d(x_new, x_old, y_old):
    """
    Linear interpolation in log space for base value pairs (x_old, y_old)
    for new x_values x_new
    :param x_old: Old x values used for interpolation
    :param y_old: Old y values used for interpolation
    :param x_new: New x values
    :retrun: y_new from liner interpolation in log space
    """
    # log of all
    logx = np.log10(x_old)
    
    logxnew = np.log10(x_new)
    
    # Avoid nan entries for yy=0 entries
    logy = np.log10(np.where(y_old <= 0, 1e-99, y_old))

    lin_interp = interp(logx, logy, logxnew)
    #lin_interp = np.interp(logxnew, logx, logy)

    return np.power(10., lin_interp)
    


def multi_response_irf_read_objects(times, detector, drm='Photopeak'):
    """
    TODO: This is very ugly. Come up with a better way.
    Function to initalize the needed responses for the given times.
    Only initalize every needed response version once! Because of memory.
    One response object needs about 1 GB of RAM...
    :param times: Times of the different sw used
    :return: list with correct response version object of the times
    """
    response_versions = []
    for time in times:
        if time==None:
            # Default latest response version
            response_versions.append(4)

        elif time<Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
            response_versions.append(0)

        elif time<Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
            response_versions.append(1)

        elif time<Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
            response_versions.append(2)

        elif time<Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
            response_versions.append(3)

        else:
            response_versions.append(4)

    responses = [None, None, None, None, None]

    response_irf_read_times = []
    for i, version in enumerate(response_versions):
        if responses[version] is None:
            #Create this response object
            if drm == "Photopeak":
                responses[version] = ResponseIRFReadPhotopeak(detector=detector,
                                                              version=version)
            else:
                responses[version] = ResponseIRFReadRMF(version=version)
                
        response_irf_read_times.append(responses[version])
    return response_irf_read_times

class ResponseIRFReadRMFNew(object):
    def __init__(self, version=None):
        """
        Object that holds the IRF's. This will be shared among all sw that use the same IRF version.
        This is done to save memory as one ResponseIRFRead object needs about 1 GB of RAM...
        :param version: Version of the IRF (from 0 to 4 or None)
        :return:
        """
        if version==0:
            irf_file = get_path_of_data_file('spi_three_irfs_database_0.hdf5')
            print('Using the irfs that are valid between Start'\
                  ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')

        elif version==1:
            irf_file = get_path_of_data_file('spi_three_irfs_database_1.hdf5')
            print('Using the irfs that are valid between 03/07/06 06:00:00'\
                  ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif version==2:
            irf_file = get_path_of_data_file('spi_three_irfs_database_2.hdf5')
            print('Using the irfs that are valid between 04/07/17 08:20:06'\
                  ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif version==3:
            irf_file = get_path_of_data_file('spi_three_irfs_database_3.hdf5')
            print('Using the irfs that are valid between 09/02/19 09:59:57'\
                  ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            print('Using the irfs that are valid between 10/05/27 12:45:00'\
                  ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')

        self._energies_database = irf_database['energies'][()]

        self._ebounds = self._energies_database
        self._ene_min = self._energies_database[:-1]
        self._ene_max = self._energies_database[1:]

        irf_data = irf_database['irfs']

        self._irfs = irf_data[()]

        self._irfs_photopeak = self._irfs[:,:,:,:,0]
        self._irfs_nonphoto_1 = self._irfs[:,:,:,:,1]
        self._irfs_nonphoto_2 = self._irfs[:,:,:,:,2]

        del self._irfs

        self._irf_xmin = irf_data.attrs['irf_xmin']
        self._irf_ymin = irf_data.attrs['irf_ymin']
        self._irf_xbin = irf_data.attrs['irf_xbin']
        self._irf_ybin = irf_data.attrs['irf_ybin']
        self._irf_nx = irf_data.attrs['nx']
        self._irf_ny = irf_data.attrs['ny']

        irf_database.close()

        self._n_dets = self._irfs_photopeak.shape[1]

        self._ebounds_rmf_2_base, self._rmf_2_base = load_rmf_non_ph_1()
        self._ebounds_rmf_3_base, self._rmf_3_base = load_rmf_non_ph_2()


class ResponseIRFReadRMF(object):
    def __init__(self, detector, version=None):
        """
        Object that holds the IRF's. This will be shared among all sw that use the same IRF version.
        This is done to save memory as one ResponseIRFRead object needs about 1 GB of RAM...
        :param version: Version of the IRF (from 0 to 4 or None)
        :return:
        """
        if version==0:
            irf_file = get_path_of_data_file('spi_three_irfs_database_0.hdf5')
            print('Using the irfs that are valid between Start'\
                  ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')
            
        elif version==1:
            irf_file = get_path_of_data_file('spi_three_irfs_database_1.hdf5')
            print('Using the irfs that are valid between 03/07/06 06:00:00'\
                  ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif version==2:
            irf_file = get_path_of_data_file('spi_three_irfs_database_2.hdf5')
            print('Using the irfs that are valid between 04/07/17 08:20:06'\
                  ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif version==3:
            irf_file = get_path_of_data_file('spi_three_irfs_database_3.hdf5')
            print('Using the irfs that are valid between 09/02/19 09:59:57'\
                  ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            print('Using the irfs that are valid between 10/05/27 12:45:00'\
                  ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')

        self._energies_database = irf_database['energies'][()]

        self._ebounds = self._energies_database
        self._ene_min = self._energies_database[:-1]
        self._ene_max = self._energies_database[1:]
        
        irf_data = irf_database['irfs']

        self._irfs = irf_data[()]

        self._irfs_photopeak = self._irfs[:,detector,:,:,0]
        self._irfs_nonphoto_1 = self._irfs[:,detector,:,:,1]
        self._irfs_nonphoto_2 = self._irfs[:,detector,:,:,2]

        del self._irfs

        self._irf_xmin = irf_data.attrs['irf_xmin']
        self._irf_ymin = irf_data.attrs['irf_ymin']
        self._irf_xbin = irf_data.attrs['irf_xbin']
        self._irf_ybin = irf_data.attrs['irf_ybin']
        self._irf_nx = irf_data.attrs['nx']
        self._irf_ny = irf_data.attrs['ny']
        
        irf_database.close()

        self._n_dets = self._irfs_photopeak.shape[1]

        self._ebounds_rmf_2_base, self._rmf_2_base = load_rmf_non_ph_1()
        self._ebounds_rmf_3_base, self._rmf_3_base = load_rmf_non_ph_2()
    
class ResponseIRFReadPhotopeak(object):
    def __init__(self, detector, version=None):
        """
        Object that holds the IRF's. This will be shared among all sw that use the same IRF version.
        This is done to save memory as one ResponseIRFRead object needs about 1 GB of RAM...
        :param version: Version of the IRF (from 0 to 4 or None)
        :return:
        """
        if version == 0:
            irf_file = get_path_of_data_file('spi_three_irfs_database_0.hdf5')
            #print('Using the irfs that are valid between Start'\
            #      ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')
            
        elif version == 1:
            irf_file = get_path_of_data_file('spi_three_irfs_database_1.hdf5')
            #print('Using the irfs that are valid between 03/07/06 06:00:00'\
            #      ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif version == 2:
            irf_file = get_path_of_data_file('spi_three_irfs_database_2.hdf5')
            #print('Using the irfs that are valid between 04/07/17 08:20:06'\
            #      ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif version == 3:
            irf_file = get_path_of_data_file('spi_three_irfs_database_3.hdf5')
            #print('Using the irfs that are valid between 09/02/19 09:59:57'\
            #      ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            #print('Using the irfs that are valid between 10/05/27 12:45:00'\
            #      ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')

        self._energies_database = irf_database['energies'][()]

        self._ebounds = self._energies_database
        self._ene_min = self._energies_database[:-1]
        self._ene_max = self._energies_database[1:]
        
        irf_data = irf_database['irfs']

        #self._irfs = irf_data[()]
        #print(detector)
        self._irfs_photopeak = irf_data[:, detector, :, :, 0]

        #del self._irfs
        
        self._irf_xmin = irf_data.attrs['irf_xmin']
        self._irf_ymin = irf_data.attrs['irf_ymin']
        self._irf_xbin = irf_data.attrs['irf_xbin']
        self._irf_ybin = irf_data.attrs['irf_ybin']
        self._irf_nx = irf_data.attrs['nx']
        self._irf_ny = irf_data.attrs['ny']
        
        irf_database.close()

        self._n_dets = self._irfs_photopeak.shape[1]

class ResponseIRFReadPhotopeakNew(object):
    def __init__(self, version=None):
        """
        Object that holds the IRF's. This will be shared among all sw that use the same IRF version.
        This is done to save memory as one ResponseIRFRead object needs about 1 GB of RAM...
        :param version: Version of the IRF (from 0 to 4 or None)
        :return:
        """
        if version == 0:
            irf_file = get_path_of_data_file('spi_three_irfs_database_0.hdf5')
            #print('Using the irfs that are valid between Start'\
            #      ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')

        elif version == 1:
            irf_file = get_path_of_data_file('spi_three_irfs_database_1.hdf5')
            #print('Using the irfs that are valid between 03/07/06 06:00:00'\
            #      ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif version == 2:
            irf_file = get_path_of_data_file('spi_three_irfs_database_2.hdf5')
            #print('Using the irfs that are valid between 04/07/17 08:20:06'\
            #      ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif version == 3:
            irf_file = get_path_of_data_file('spi_three_irfs_database_3.hdf5')
            #print('Using the irfs that are valid between 09/02/19 09:59:57'\
            #      ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            #print('Using the irfs that are valid between 10/05/27 12:45:00'\
            #      ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')

        self._energies_database = irf_database['energies'][()]

        self._ebounds = self._energies_database
        self._ene_min = self._energies_database[:-1]
        self._ene_max = self._energies_database[1:]

        irf_data = irf_database['irfs']

        #self._irfs = irf_data[()]
        #print(detector)
        self._irfs_photopeak = irf_data[:, :, :, :, 0]

        #del self._irfs

        self._irf_xmin = irf_data.attrs['irf_xmin']
        self._irf_ymin = irf_data.attrs['irf_ymin']
        self._irf_xbin = irf_data.attrs['irf_xbin']
        self._irf_ybin = irf_data.attrs['irf_ybin']
        self._irf_nx = irf_data.attrs['nx']
        self._irf_ny = irf_data.attrs['ny']

        irf_database.close()

        self._n_dets = self._irfs_photopeak.shape[1]


        
class Response(object):
    def __init__(self, ebounds=None, response_irf_read_object=None, sc_matrix=None, det=None):
        """FIXME! briefly describe function
        :param ebounds: User defined ebins for binned effective area
        :param response_irf_read_object: Object that holds the read in irf values
        :returns: 
        :rtype: 
        """
        
        self._irf_ob = response_irf_read_object
        self._ebounds = self._irf_ob._ebounds
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)
        self._sc_matrix = sc_matrix
        self._psd_bins = self._get_psd_bins(ebounds)
        self._det = det

    def _get_psd_bins(self, ebounds):
        """
        Get which ebins are in the electronic noise range
        :param ebounds: Ebounds of Ebins
        :return: mask with the psd bins set to true
        """
        psd_low_end = 1400
        psd_high_end = 1700
        psd_bins = np.zeros(len(ebounds)-1, dtype=bool)
        for i, (e_min, e_max) in enumerate(zip(ebounds[:-1],ebounds[1:])):
            if e_min<=psd_low_end:
                if e_max>=psd_low_end:
                    psd_bins[i] = True
            elif e_min<=psd_high_end:
                psd_bins[i] = True
        return psd_bins

    def set_binned_data_energy_bounds(self, ebounds):
        """
        Change the energy bins for the binned effective_area
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :return:
        """

        if not np.array_equal(ebounds, self._ebounds):

            #print('You have changed the energy boundaries for the binned effective_area calculation in the further calculations!')
            self._ene_min = ebounds[:-1]
            self._ene_max = ebounds[1:]
            self._ebounds = ebounds
    
    def get_xy_pos(self, azimuth, zenith):
        """
        Get xy position (in SPI simulation) for given azimuth and zenith
        :param azimuth:
        :param zenith:
        :returns: 
        """

        return _get_xy_pos(azimuth, zenith, self._irf_ob._irf_xmin, self._irf_ob._irf_ymin, self._irf_ob._irf_xbin, self._irf_ob._irf_ybin)
    
    def set_location(self, ra, dec):
        """
        Calculate the weighted irfs for the three event types for a given position
        :param azimuth: Azimuth position in sat frame
        :param zenith: Zenith position in sat frame
        :returns:
        """

        # Transform ra, dec from icrs to spi frame
        azimuth, zenith = _transform_icrs_to_spi(ra,
                                                 dec,
                                                 self._sc_matrix)


        self._weighted_irfs(np.deg2rad(azimuth),
                            np.deg2rad(zenith))

        self._recalculate_response()

    def _weighted_irfs(self, az, zen):

        raise NotImplementedError("Must be implented in child class.")

    def _recalculate_response(self):

        raise NotImplementedError("Must be implented in child class.")

    def _get_irf_weights(self, x_pos, y_pos):
        """FIXME! briefly describe function

        :param x_pos: 
        :param y_pos: 
        :returns: 
        :rtype: 

        """


        # get the four nearest neighbors
        ix_left = np.floor(x_pos) if (x_pos >= 0.0) else np.floor(x_pos) - 1
        iy_low = np.floor(y_pos) if (y_pos >= 0.0) else np.floor(y_pos) - 1

        ix_right = ix_left + 1
        iy_up = iy_low + 1

        wgt_right = float(x_pos - ix_left)
        wgt_up = float(y_pos - iy_low)



        
        # pre set the weights
        wgt = np.zeros(4)


        if ix_left < 0.:

            if ix_right < 0.:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

            else:

                ix_left = ix_right
                wgt_left = 0.5
                wgt_right = 0.5

        elif ix_right >= self._irf_ob._irf_nx:

            if ix_left >= self._irf_ob._irf_nx:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

            else:

                ix_right = ix_left
                wgt_left = 0.5
                wgt_right = 0.5

        else:

            wgt_left = 1. - wgt_right

        if iy_low < 0:
            if iy_up < 0:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]
            else:
                iy_low = iy_up
                wgt_up = 0.5
                wgt_low = 0.5

        elif iy_up >= self._irf_ob._irf_ny:

            if iy_low >= self._irf_ob._irf_ny:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

            else:

                iy_up = iy_low
                wgt_up = 0.5
                wgt_low = 0.5

        else:

            wgt_low = 1. - wgt_up

        wgt[0] = wgt_left * wgt_low
        wgt[1] = wgt_right * wgt_low
        wgt[2] = wgt_left * wgt_up
        wgt[3] = wgt_right * wgt_up

        out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

        return wgt, out[0], out[1]


        

        


    
    def get_irf_weights_vector(self, x_pos, y_pos):

        raise NotImplementedError('Cannot do this yet')

        idx_x_neg = x_pos < 0.
        idx_y_neg = y_pos < 0.

        ix_left = x_pos
        iy_low = y_pos

        ix_left[idx_x_neg] -= 1
        iy_low[idx_y_neg] -= 1

        ix_right = ix_left + 1
        iy_up = iy_low + 1
        wgt_right = x_pos - ix_left
        wgt_up = y_pos - iy_low

        wgt_left = 1. - wgt_right
        wgt_low = 1. - wgt_up

        ############################################

        selection = (ix_left < 0.) & (ix_right >= 0.)

        wgt_left[selection] = 0.5
        wgt_right[selection] = 0.5

        ix_left[selection] = ix_right[selection]

        selection = (ix_right >= self._irf_nx) & (ix_left < self._irf_nx)

        wgt_left[selection] = 0.5
        wgt_right[selection] = 0.5

        ix_right[selection] = ix_left[selection]

        selection = (iy_low < 0) & (iy_up >= 0)

        iy_low[selection] = iy_up[selection]
        wgt_up[selection] = 0.5
        wgt_low[selection] = 0.5

        selection = (iy_up >= self._irf_ny) & (iy_low < self._irf_ny)

        iy_up[selection] = iy_low[selection]
        wgt_up[selection] = 0.5
        wgt_low[selection] = 0.5

        #         inx[0] = int(ix_left + iy_low * self._irf_nx)
        #         inx[1] = int(ix_right + iy_low * self._irf_nx)
        #         inx[2] = int(ix_left + iy_up * self._irf_nx)
        #         inx[3] = int(ix_right + iy_up * self._irf_nx)

        left_low = [int(ix_left), int(iy_low)]
        right_low = [int(ix_right), int(iy_low)]
        left_up = [int(ix_left), int(iy_up)]
        right_up = [int(ix_right), int(iy_up)]

        wgt[0] = wgt_left * wgt_low
        wgt[1] = wgt_right * wgt_low
        wgt[2] = wgt_left * wgt_up
        wgt[3] = wgt_right * wgt_up

        return inx, wgt,

    @property
    def irfs(self):

        return self._irfs

    @property
    def energies_database(self):
        return self._energies_database

    @property
    def ebounds(self):
        return self._ebounds

    @property
    def ene_min(self):
        return self._ene_min

    @property
    def ene_max(self):
        return self._ene_max
    
    @property
    def rod(self):
        """
        Ensure that you know what you are doing.

        :return: Roland
        """
        return HTML(filename=get_path_of_data_file('roland.html'))

class ResponseRMFNew(Response):

    def __init__(self,
                 ebounds=None,
                 response_irf_read_object=None,
                 sc_matrix=None, det=None,
                 fixed_rsp_matrix=None):
        """
        Init Response object with total RMF used
        :param ebound: Ebounds of Ebins
        :param response_irf_read_object: Object that holds the read in irf values
        :return:
        """
        super(ResponseRMFNew, self).__init__(ebounds=ebounds,
                                             response_irf_read_object=response_irf_read_object,
                                             sc_matrix=sc_matrix,
                                             det=det)
        if fixed_rsp_matrix is None:
            assert isinstance(response_irf_read_object, ResponseIRFReadRMFNew)

            idx = np.array([], dtype=int)
            for el, eh in zip(ebounds[:-1], ebounds[1:]):

                assert (el,eh) in zip(response_irf_read_object._ebounds_rmf_2_base[:-1],
                                      response_irf_read_object._ebounds_rmf_2_base[1:]), \
                                      "Only works for the base ebounds like in the"\
                                      f" original rmf files. {el}-{eh} is not part of this."

                idx = np.append(idx, np.argwhere(el==response_irf_read_object._ebounds_rmf_2_base)[0,0])
            #idx = np.append(idx, idx[-1]+1)

            self._mat1inter = interpolate.interp1d(self._irf_ob._energies_database,
                                                   self._irf_ob._rmf_2_base,
                                                   fill_value="extrapolate",
                                                   axis=0)(ebounds)[:,idx].T

            self._mat2inter = interpolate.interp1d(self._irf_ob._energies_database,
                                                   self._irf_ob._rmf_3_base,
                                                   fill_value="extrapolate",
                                                   axis=0)(ebounds)[:,idx].T
            self._given_rsp_mat = False
        else:
            self._given_rsp_mat = True
            self._rsp_matrix = fixed_rsp_matrix

        self._monte_carlo_energies = self._ebounds


    @classmethod
    def from_config(cls, config, det, rsp_read_obj, fixed_rsp_matrix=None):
        """
        Construct the Response object from an given config file.
        """
        if not isinstance(config, dict):

            if isinstance(config, Config):
                configuration = config
            else:
                # Assume this is a file name
                configuration_file = sanitize_filename(config)

                assert os.path.exists(config), "Configuration file %s does not exist" % configuration_file

                # Read the configuration
                with open(configuration_file) as f:

                    configuration = yaml.safe_load(f)

        else:

            # Configuration is a dictionary. Nothing to do
            configuration = config

        # Construct ebounds

        # Binned or unbinned analysis?
        binned = configuration['Energy_binned']
        if binned:

            # Set ebounds of energy bins
            ebounds = np.array(configuration['Ebounds'])

            # If no ebounds are given raise Assertion
            assert ebounds is not None, "Please give bounds for the energy bins"

            # Construct final energy bins (make sure to make extra echans for the electronic noise energy range)
            ebounds, _ = construct_energy_bins(ebounds)
        else:
            raise NotImplementedError('Unbinned analysis not implemented!')

        # Get time of GRB
        time_of_grb = configuration['Time_of_GRB_UTC']
        time = datetime.strptime(time_of_grb, '%y%m%d %H%M%S')
        time = Time(time)
        if time < Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
            version = 0

        elif time < Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
            version = 1

        elif time < Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
            version = 2

        elif time < Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
            version = 3

        else:
            version = 4

        # Load correct base irf response read object
        #rsp_read_obj = ResponseIRFReadRMFNew(version)

        # Construct sc_matrix of this sw
        pointing_id = find_needed_ids(time)

        try:
            # Get the data from the afs server
            get_files_afs(pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(pointing_id)

        geometry_file_path = os.path.join(get_path_of_external_data_dir(),
                                          'pointing_data',
                                          pointing_id,
                                          'sc_orbit_param.fits.gz')

        pointing_object = SPIPointing(geometry_file_path)
        sc_matrix = _construct_sc_matrix(**pointing_object.sc_points[10])

        # Init Response class
        return cls(
            ebounds=ebounds,
            response_irf_read_object=rsp_read_obj,
            sc_matrix=sc_matrix,
            det=det,
            fixed_rsp_matrix=fixed_rsp_matrix
        )

    @classmethod
    def from_pointing(cls,
                      pointing_id,
                      det,
                      ebounds,
                      rsp_read_obj,
                      fixed_rsp_matrix=None):
        """
        Construct the Response object from an given config file.
        """
        try:
            # Get the data from the afs server
            get_files_afs(pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(pointing_id)

        geometry_file_path = os.path.join(get_path_of_external_data_dir(),
                                          'pointing_data',
                                          pointing_id,
                                          'sc_orbit_param.fits.gz')

        pointing_object = SPIPointing(geometry_file_path)
        sc_matrix = _construct_sc_matrix(**pointing_object.sc_points[10])

        # Init Response class
        return cls(
            ebounds=ebounds,
            response_irf_read_object=rsp_read_obj,
            sc_matrix=sc_matrix,
            det=det,
            fixed_rsp_matrix=fixed_rsp_matrix
        )

    def _recalculate_response(self):
        """
        Get response for a given det
        :param det: Detector ID
        :returns: Full DRM
        """
        #n_energy_bins = len(self._ebounds) - 1
        if self._given_rsp_mat:
            self._matrix = self._rsp_matrix
        else:
            ebins = np.empty((len(self._ene_min), 2))
            ph = np.empty_like(ebins)
            nonph1 = np.empty_like(ebins)
            nonph2 = np.empty_like(ebins)

            ebins[:, 0] = self._ene_min
            ebins[:, 1] = self._ene_max

            interph = log_interp1d(self._ebounds,
                                   self._irf_ob._energies_database,
                                   self._weighted_irf_ph)
            inter1 = log_interp1d(self._ebounds,
                                  self._irf_ob._energies_database,
                                  self._weighted_irf_nonph_1)
            inter2 = log_interp1d(self._ebounds,
                                  self._irf_ob._energies_database,
                                  self._weighted_irf_nonph_2)

            ph[:, 0] = interph[:-1]
            ph[:, 1] = interph[1:]
            #nonph1[:, 0] = inter1[:-1]
            #nonph1[:, 1] = inter1[1:]
            #nonph2[:, 0] = inter2[:-1]
            #nonph2[:, 1] = inter2[1:]

            integrate_ph = trapz(ph, ebins)/(self._ene_max-self._ene_min)
            #self._integrate_nonph1 = trapz(nonph1, ebins)#/(self._ene_max-self._ene_min) cancels with factor below
            #self._integrate_nonph2 = trapz(nonph2, ebins)#/(self._ene_max-self._ene_min)

            mat1 = (inter1*self._mat1inter).T
            mat2 = (inter2*self._mat2inter).T

            # Trapz integrate the non-psd matrix
            self._transpose_matrix = (mat1[1:]+mat2[1:]+mat1[:-1]+mat2[:-1])/2.

            # Add photopeak
            for i in range(len(self._transpose_matrix)):
                self._transpose_matrix[i,i] += integrate_ph[i]

            self._matrix = self._transpose_matrix.T

    def _weighted_irfs(self, azimuth, zenith):
        """
        Calculate the weighted irfs for the three event types for a given position
        :param azimuth: Azimuth position in sat frame
        :param zenith: Zenith position in sat frame
        :returns:
        """

        # get the x,y position on the grid
        x, y = self.get_xy_pos(azimuth, zenith)

        # compute the weights between the grids
        wgt, xx, yy = self._get_irf_weights(x, y)


        # If outside of the response pattern set response to zero
        try:
            # select these points on the grid and weight them together
            self._weighted_irf_ph = self._irf_ob._irfs_photopeak[:,self._det, xx, yy].dot(wgt)
            self._weighted_irf_nonph_1 = self._irf_ob._irfs_nonphoto_1[:,self._det,xx,yy].dot(wgt)
            self._weighted_irf_nonph_2 = self._irf_ob._irfs_nonphoto_2[:,self._det,xx,yy].dot(wgt)
        except IndexError:
            self._weighted_irf_ph = np.zeros_like(self._irf_ob._irfs_photopeak[:,self._det,20,20])
            self._weighted_irf_nonph_1 = np.zeros_like(self._irf_ob._irfs_nonphoto_1[:,self._det,20,20])
            self._weighted_irf_nonph_2 = np.zeros_like(self._irf_ob._irfs_nonphoto_2[:,self._det,20,20])

    @property
    def matrix(self):
        return self._matrix

    @property
    def transpose_matrix(self):
        return self._transpose_matrix

    @property
    def ebounds(self):
        return self._ebounds

    @property
    def monte_carlo_energies(self):
        return self._monte_carlo_energies

try:
    from threeML.utils.OGIP.response import InstrumentResponse

except:
    from responsum import InstrumentResponse
class SPIDRM(InstrumentResponse):
    def __init__(self, drm_generator, ra, dec):
        self._drm_generator = drm_generator

        self._drm_generator.set_location(ra, dec)
        self._min_dist = np.deg2rad(.5)


        super(SPIDRM, self).__init__(
            self._drm_generator.matrix,
            self._drm_generator.ebounds,
            self._drm_generator.monte_carlo_energies,
        )

    def set_location(self, ra, dec, cache=False):
        """
        Set the source location
        :param ra:
        :param dec:
        :return:
        """
        self._drm_generator.set_location(ra, dec)

        self._matrix = self._drm_generator.matrix
        self._matrix_transpose = self._matrix.T

class ResponseRMF(Response):

    def __init__(self, ebounds=None, response_irf_read_object=None, sc_matrix=None):
        """
        Init Response object with total RMF used
        :param ebound: Ebounds of Ebins
        :param response_irf_read_object: Object that holds the read in irf values
        :return:
        """
        assert isinstance(response_irf_read_object, ResponseIRFReadRMF)
        #assert np.all(np.equal(ebounds, response_irf_read_object._ebounds_rmf_2_base)), "Only works for the base ebounds like in the original rmf files"
        super(ResponseRMF, self).__init__(ebounds, response_irf_read_object, sc_matrix)
        
    def set_binned_data_energy_bounds(self, ebounds):
        """
        Change the energy bins for the binned effective_area
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :return:
        """

        if not np.array_equal(ebounds, self._ebounds):
            
            print('You have changed the energy boundaries for the binned effective_area calculation in the further calculations!')
            self._ene_min = ebounds[:-1]
            self._ene_max = ebounds[1:]
            self._ebounds = ebounds

            self._rmf2 = self._rebin_rmfs(self._ebounds, self._irf_ob._ebounds_rmf_2_base, self._irf_ob._rmf_2_base)
            self._rmf3 = self._rebin_rmfs(self._ebounds, self._irf_ob._ebounds_rmf_3_base, self._irf_ob._rmf_3_base)

    def _rebin_rmfs(self, new_ebins, old_ebins, old_rmf):
        """
        Rebin base rmf's to new ebins
        :param new_ebins: New energy bins
        :param old_ebins: Ebins in base rmf
        :param old_rmf: base rmf
        """
        def low_index_bracket(value, array):
            for i in range(len(array)-1):
                if value>=array[i] and value<array[i+1]:
                    return i
            return -1

        def bilinint(x, y, matrix, x0, y0):

            x1_ind, x2_ind = np.abs(x-x0).argsort()[:2]
            y1_ind, y2_ind = np.abs(y-y0).argsort()[:2]

            x1 = x[x1_ind]
            x2 = x[x2_ind]
            y1 = x[y1_ind]
            y2 = x[y2_ind]

            denom = (x2-x1)*(y2-y1)
            value = (matrix[x1_ind, y1_ind]*(x2-x0)*(y2-y0)+ matrix[x2_ind, y1_ind]*(x0-x1)*(y2-y0)+matrix[x1_ind, y2_ind]*(x2-x0)*(y0-y1)+matrix[x2_ind, y2_ind]*(x0-x1)*(y0-y1))/denom
            return value

        
        min_e_old = old_ebins[:-1]
        max_e_old = old_ebins[1:]

        min_e_new = new_ebins[:-1]
        max_e_new = new_ebins[1:]
    
        x_old = np.power(10., (np.log10(min_e_old)+np.log10(max_e_old))/2.)
        y_old = np.power(10., (np.log10(min_e_old)+np.log10(max_e_old))/2.)

        matrix = (np.copy(old_rmf)/(max_e_old-min_e_old))
        diag_old = np.diag(np.copy(old_rmf))
        x_new_temp = np.power(10., (np.log10(min_e_new)+np.log10(max_e_new))/2.)
        y_new_temp = np.power(10., (np.log10(min_e_new)+np.log10(max_e_new))/2.)

        IntMatrix = np.zeros((len(x_new_temp),len(y_new_temp)))
        for i in range(len(IntMatrix)):

            j = low_index_bracket(x_new_temp[i], x_old)

            if j!=0:
                matrix[j,j] = matrix[j, j-1]
                matrix[j+1, j+1] = matrix[j+1,j]
            for k in range(i):
                IntMatrix[i,k] = bilinint(x_old, y_old, matrix, x_new_temp[i], y_new_temp[k])
            IntMatrix[i] *= max_e_new-min_e_new
    
            # Diag interpolation
            IntMatrix[i,i] = diag_old[j]*(1-(x_new_temp[i]-x_old[j])/(x_old[j+1]-x_old[j])) + (x_new_temp[i]-x_old[j])/(x_old[j+1]-x_old[j])*diag_old[j+1]

        return IntMatrix

    def _get_response_det(self, det):
        """
        Get response for a given det
        :param det: Detector ID
        :returns: Full DRM
        """
        
        n_energy_bins = len(self._ebounds) - 1
        
        ebins = np.empty((len(self._ene_min), 2))

        # photopeak integrated irfs
        ebins[:,0] = self._ene_min
        ebins[:,1] = self._ene_max

        ph_irfs = np.empty_like(ebins)

        # Get interpolated irfs
        inter = log_interp1d(self._ebounds, self._irf_ob._energies_database, self._weighted_irf_ph[:, det])
        
        ph_irfs[:,0] = inter[:-1]
        ph_irfs[:,1] = inter[1:]

        ph_irfs_int = trapz(ph_irfs, ebins)/(self._ene_max-self._ene_min)

        # RMF1 and RMF2 matrix
        nonph1_irfs = np.empty_like(ebins)
        inter = log_interp1d(self._ebounds, self._irf_ob._energies_database, self._weighted_irf_nonph_1[:, det]) 
        
        nonph1_irfs[:,0] = inter[:-1]
        nonph1_irfs[:,1] = inter[1:]
        
        nonph1_irfs_int = trapz(nonph1_irfs, ebins)/(self._ene_max-self._ene_min)

        nonph2_irfs = np.empty_like(ebins)
        inter = log_interp1d(self._ebounds, self._irf_ob._energies_database, self._weighted_irf_nonph_2[:, det]) 
        
        nonph2_irfs[:,0] = inter[:-1]
        nonph2_irfs[:,1] = inter[1:]
        
        nonph2_irfs_int = trapz(nonph2_irfs, ebins)/(self._ene_max-self._ene_min)
        

        # Build DRM
        drm = ph_irfs_int*np.identity(len(self._ene_min)) + \
            nonph1_irfs_int*self._rmf2.T + nonph1_irfs_int*self._rmf3.T
        
        return drm

    def _weighted_irfs(self, azimuth, zenith):
        """
        Calculate the weighted irfs for the three event types for a given position
        :param azimuth: Azimuth position in sat frame 
        :param zenith: Zenith position in sat frame
        :returns:
        """

        # get the x,y position on the grid
        x, y = self.get_xy_pos(azimuth, zenith)

        # compute the weights between the grids
        wgt, xx, yy = self._get_irf_weights(x, y)


        # If outside of the response pattern set response to zero
        try:
            # select these points on the grid and weight them together
            self._weighted_irf_ph = self._irf_ob._irfs_photopeak[..., xx, yy].dot(wgt)
            self._weighted_irf_nonph_1 = self._irf_ob._irfs_nonphoto_1[...,xx,yy].dot(wgt)
            self._weighted_irf_nonph_2 = self._irf_ob._irfs_nonphoto_2[...,xx,yy].dot(wgt)
        except IndexError:
            self._weighted_irf_ph = np.zeros_like(self._irf_ob._irfs_photopeak[...,20,20])
            self._weighted_irf_nonph_1 = np.zeros_like(self._irf_ob._irfs_nonphoto_1[...,20,20])
            self._weighted_irf_nonph_2 = np.zeros_like(self._irf_ob._irfs_nonphoto_2[...,20,20])
            
        #return weighted_irf_ph, weighted_irf_nonph_1, weighted_irf_nonph_2

    
    #def _interpolated_irfs(self, azimuth, zenith):
    #    """
    #    Return three lists with the interploated irf curves for
    #    each detector#

    #     :param azimuth: 
    #    :param zenith: 
    #    :returns: 
    #    :rtype: 

    #     """

    #    weighted_irf_ph, weighted_irf_nonph_1, weighted_irf_nonph_2 = self._weighted_irfs(azimuth, zenith)

        
        
    #    interpolated_irfs_ph = []
    #    interpolated_irfs_nonph1 = []
    #    interpolated_irfs_nonph2 = []
        
    #    for det_number in range(self._n_dets):

            #tmp = interpolate.interp1d(self._energies, weighted_irf[:, det_number])
            
    #        tmp = log_interp1d(self._energies_database, weighted_irf_ph[:, det_number])
    #        tmp2 = log_interp1d(self._energies_database, weighted_irf_nonph_1[:, det_number])
    #        tmp3 = log_interp1d(self._energies_database, weighted_irf_nonph_2[:, det_number])
            
    #        interpolated_irfs_ph.append(tmp)
    #        interpolated_irfs_nonph1.append(tmp2)
    #        interpolated_irfs_nonph2.append(tmp3)
            
    #    self._current_interpolated_irfs_ph = interpolated_irfs_ph
    #    self._current_interpolated_irfs_nonph1 = interpolated_irfs_nonph1
    #    self._current_interpolated_irfs_nonph2 = interpolated_irfs_nonph2


class ResponsePhotopeak(Response):

    def __init__(self, ebounds=None, response_irf_read_object=None, sc_matrix=None, det=None):
        """
        Init Response object with only Photopeak effective area used
        :param ebound: Ebounds of Ebins
        :param response_irf_read_object: Object that holds the read in irf values
        :return:
        """
        super(ResponsePhotopeak, self).__init__(ebounds, response_irf_read_object, sc_matrix, det)

    @classmethod
    def from_config(cls, config, det, rsp_read_obj):
        """
        Construct the Response object from an given config file.
        """
        if not isinstance(config, dict):

            if isinstance(config, Config):
                configuration = config
            else:
                # Assume this is a file name
                configuration_file = sanitize_filename(config)

                assert os.path.exists(config), "Configuration file %s does not exist" % configuration_file

                # Read the configuration
                with open(configuration_file) as f:

                    configuration = yaml.safe_load(f)

        else:

            # Configuration is a dictionary. Nothing to do
            configuration = config

        # Construct ebounds

        # Binned or unbinned analysis?
        binned = configuration['Energy_binned']
        if binned:

            # Set ebounds of energy bins
            ebounds = np.array(configuration['Ebounds'])

            # If no ebounds are given raise Assertion
            assert ebounds is not None, "Please give bounds for the energy bins"

            # Construct final energy bins (make sure to make extra echans for the electronic noise energy range)
            ebounds, _ = construct_energy_bins(ebounds)
        else:
            raise NotImplementedError('Unbinned analysis not implemented!')
        # Get time of GRB
        time_of_grb = configuration['Time_of_GRB_UTC']
        time = datetime.strptime(time_of_grb, '%y%m%d %H%M%S')
        time = Time(time)
        if time < Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
            version = 0

        elif time < Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
            version = 1

        elif time < Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
            version = 2

        elif time < Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
            version = 3

        else:
            version = 4

        # Load correct base irf response read object
        #rsp_read_obj = ResponseIRFReadPhotopeak(det, version)

        # Construct sc_matrix of this sw
        pointing_id = find_needed_ids(time)

        try:
            # Get the data from the afs server
            get_files_afs(pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(pointing_id)
            
        geometry_file_path = os.path.join(get_path_of_external_data_dir(),
                                          'pointing_data',
                                          pointing_id,
                                          'sc_orbit_param.fits.gz')

        pointing_object = SPIPointing(geometry_file_path)
        sc_matrix = _construct_sc_matrix(**pointing_object.sc_points[10])

        # Init Response class
        return cls(
            ebounds=ebounds,
            response_irf_read_object=rsp_read_obj,
            sc_matrix=sc_matrix,
            det=det
        )

    @classmethod
    def from_pointing(cls,
                      pointing_id,
                      det,
                      ebounds,
                      rsp_read_obj):
        """
        Construct the Response object from an given config file.
        """
        try:
            # Get the data from the afs server
            get_files_afs(pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(pointing_id)

        geometry_file_path = os.path.join(get_path_of_external_data_dir(),
                                          'pointing_data',
                                          pointing_id,
                                          'sc_orbit_param.fits.gz')

        pointing_object = SPIPointing(geometry_file_path)
        sc_matrix = _construct_sc_matrix(**pointing_object.sc_points[10])

        # Init Response class
        return cls(
            ebounds=ebounds,
            response_irf_read_object=rsp_read_obj,
            sc_matrix=sc_matrix,
            det=det,
        )

    def _recalculate_response(self):
        """
        Get response for a given det
        :param det: Detector ID
        :returns: Full DRM
        """
        #n_energy_bins = len(self._ebounds) - 1
        
        ebins = np.empty((len(self._ene_min), 2))
        eff_area = np.empty_like(ebins)

        ebins[:, 0] = self._ene_min
        ebins[:, 1] = self._ene_max
        
        inter = log_interp1d(self._ebounds,
                             self._irf_ob._energies_database,
                             self._weighted_irf_ph)
        
        eff_area[:, 0] = inter[:-1]
        eff_area[:, 1] = inter[1:]
        
        self._effective_area = trapz(eff_area, ebins)/(self._ene_max-self._ene_min)

    def _weighted_irfs(self, azimuth, zenith):
        """
        Calculate the weighted irfs for the three event types for a given position
        :param azimuth: Azimuth position in sat frame 
        :param zenith: Zenith position in sat frame
        :returns:
        """

        # get the x,y position on the grid
        x, y = self.get_xy_pos(azimuth, zenith)

        # compute the weights between the grids
        wgt, xx, yy = self._get_irf_weights(x, y)
        print(xx)
        print(yy)

        # If outside of the response pattern set response to zero
        try:
            # select these points on the grid and weight them together
            self._weighted_irf_ph = self._irf_ob._irfs_photopeak[:,self._det, xx, yy].dot(wgt)
            print(self._weighted_irf_ph)
        except IndexError:
            self._weighted_irf_ph = np.zeros_like(self._irf_ob._irfs_photopeak[:,self._det,20,20])

    @property
    def effective_area(self):
        return self._effective_area


        #return weighted_irf_ph

    
    #def _interpolated_irfs(self, azimuth, zenith):
    #    """
    #    Get interploated irf curves for each detector
    #    :param azimuth: azimuth in sat frame of source
    #    :param zenith: zenith in sat frame of source
    #    :returns: 
    #    """

    #    weighted_irf_ph = self._weighted_irfs(azimuth, zenith)

        
        
    #    interpolated_irfs_ph = []
        
    #    for det_number in range(self._n_dets):

            #tmp = interpolate.interp1d(self._energies, weighted_irf[:, det_number])
            
    #        tmp = log_interp1d(self._energies_database, weighted_irf_ph[:, det_number])
            
    #        interpolated_irfs_ph.append(tmp)
            
    #    self._current_interpolated_irfs_ph = interpolated_irfs_ph

@njit(fastmath=True)
def _get_xy_pos(azimuth, zenith, xmin, ymin, xbin, ybin):

    x = np.cos(azimuth)*np.cos(zenith)
    y = np.sin(azimuth)*np.cos(zenith)
    z = np.sin(zenith)

    zenith_pointing = np.arccos(x)
    azimuth_pointing = np.arctan2(z,y)
        
    x_pos = (zenith_pointing * np.cos(azimuth_pointing) - xmin) / xbin
    y_pos = (zenith_pointing * np.sin(azimuth_pointing) - ymin) / ybin

    return x_pos, y_pos


    
def _prep_out_pixels(ix_left, ix_right, iy_low, iy_up):
    
    left_low = [int(ix_left), int(iy_low)]
    right_low = [int(ix_right), int(iy_low)]
    left_up = [int(ix_left), int(iy_up)]
    right_up = [int(ix_right), int(iy_up)]
    
    out = np.array([left_low, right_low, left_up, right_up]).T

    return out
