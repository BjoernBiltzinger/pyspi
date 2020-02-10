import numpy as np
import h5py
import scipy.interpolate as interpolate
import scipy.integrate as integrate
from datetime import datetime
from astropy.time.core import Time
from pyspi.utils.rmf_base import *

from IPython.display import HTML

from pyspi.io.package_data import get_path_of_data_file




def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    # Avoid nan entries for yy=0 entries
    logy = np.log10(np.where(yy<=0, 1e-32, yy))
    lin_interp = interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

class Response(object):
    def __init__(self, ebounds=None, time=None):
        """FIXME! briefly describe function
        :param time: Time object, with the time for which the valid response should be used
        :param ebounds: User defined ebins for binned effective area
        :returns: 
        :rtype: 
        """
        
        self._load_irfs(time)
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)
                
    def get_xy_pos(self, azimuth, zenith):
        """
        FIXME! briefly describe function

        :param azimuth: 
        x = np.cos(ra_sat)*np.cos(dec_sat):param zenith: 
        :returns: 
        :rtype: 

        """
        # np.pi/2 - zenith. TODO: Check if this is corect. Only a guess at the moment!
        # zenith = np.pi/2-zenith
        x = np.cos(azimuth)*np.cos(zenith)
        y = np.sin(azimuth)*np.cos(zenith)
        z = np.sin(zenith)

        zenith_pointing = np.arccos(x)
        azimuth_pointing = np.arctan2(z,y)
        
        x_pos = (zenith_pointing * np.cos(azimuth_pointing) - self._irf_xmin) / self._irf_xbin
        y_pos = (zenith_pointing * np.sin(azimuth_pointing) - self._irf_ymin) / self._irf_ybin

        return x_pos, y_pos
        
    def set_location(self, azimuth, zenith):
        """
        Update location and get new irf values for this location
        :param azimuth: Azimuth of position in spacecraft coordinates
        :param zenith: Zenith of position in spacecraft coordinates
        """
        azimuth = np.deg2rad(azimuth)
        zenith = np.deg2rad(zenith)

        self._interpolated_irfs(azimuth, zenith)
            
    def get_response_det(self, det):
        """
        Get the response for the current position for one detector
        :param det: Detector
        """

        return self._get_response_det(det)

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

        elif ix_right >= self._irf_nx:

            if ix_left >= self._irf_nx:

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

        elif iy_up >= self._irf_ny:

            if iy_low >= self._irf_ny:

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


class ResponseRMF(Response):

    def __init__(self, ebounds=None, time=None):
        """
        Init Response object with total RMF used
        :param ebound: Ebounds of Ebins
        :param time: Time for which the response should be valid
        :return:
        """
        super(ResponseRMF, self).__init__(ebounds, time)
        
    def _load_irfs(self, time=None):
        """FIXME! briefly describe function
        :param time: Time object, with the time for which the valid response should be used
        :returns: 
        :rtype: 
        """
        
        if time==None:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            print('Using the default irfs. The ones that are valid between 10/05/27 12:45:00'\
                  ' and present (YY/MM/DD HH:MM:SS)')
            
        elif time<Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_0.hdf5')
            print('Using the irfs that are valid between Start'\
                  ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')
            
        elif time<Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_1.hdf5')
            print('Using the irfs that are valid between 03/07/06 06:00:00'\
                  ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif time<Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_2.hdf5')
            print('Using the irfs that are valid between 04/07/17 08:20:06'\
                  ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif time<Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_3.hdf5')
            print('Using the irfs that are valid between 09/02/19 09:59:57'\
                  ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            print('Using the irfs that are valid between 10/05/27 12:45:00'\
                  ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')

        self._energies_database = irf_database['energies'].value

        self._ebounds = self._energies_database
        self._ene_min = self._energies_database[:-1]
        self._ene_max = self._energies_database[1:]
        
        irf_data = irf_database['irfs']

        self._irfs = irf_data[()]

        self._irfs_photopeak = self._irfs[...,0]
        self._irfs_nonphoto_1 = self._irfs[...,1]
        self._irfs_nonphoto_2 = self._irfs[...,2]

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

            self._rmf2 = self._rebin_rmfs(self._ebounds, self._ebounds_rmf_2_base, self._rmf_2_base)
            self._rmf3 = self._rebin_rmfs(self._ebounds, self._ebounds_rmf_3_base, self._rmf_3_base)

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

        interpolated_irfs_ph = self._current_interpolated_irfs_ph[det]
        interpolated_irfs_nonph1 = self._current_interpolated_irfs_nonph1[det]
        interpolated_irfs_nonph2 = self._current_interpolated_irfs_nonph2[det]
        

        n_energy_bins = len(self._ebounds) - 1
        
        ebins = np.empty((len(self._ene_min), 2))

        # photopeak integrated irfs
        ebins[:,0] = self._ene_min
        ebins[:,1] = self._ene_max

        ph_irfs = np.empty_like(ebins)
        inter = interpolated_irfs_ph(self._ebounds)
        
        ph_irfs[:,0] = inter[:-1]
        ph_irfs[:,1] = inter[1:]
        ph_irfs_int = integrate.trapz(ph_irfs, ebins, axis=1)/(self._ene_max-self._ene_min)

        # RMF1 and RMF2 matrix
        nonph1_irfs = np.empty_like(ebins)
        inter = interpolated_irfs_nonph1(self._ebounds)
        
        nonph1_irfs[:,0] = inter[:-1]
        nonph1_irfs[:,1] = inter[1:]
        nonph1_irfs_int = integrate.trapz(nonph1_irfs, ebins, axis=1)/(self._ene_max-self._ene_min)
        
        nonph2_irfs = np.empty_like(ebins)
        inter = interpolated_irfs_nonph2(self._ebounds)
        
        nonph2_irfs[:,0] = inter[:-1]
        nonph2_irfs[:,1] = inter[1:]
        nonph2_irfs_int = integrate.trapz(nonph2_irfs, ebins, axis=1)/(self._ene_max-self._ene_min)

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
            weighted_irf_ph = self._irfs_photopeak[..., xx, yy].dot(wgt)
            weighted_irf_nonph_1 = self._irfs_nonphoto_1[...,xx,yy].dot(wgt)
            weighted_irf_nonph_2 = self._irfs_nonphoto_2[...,xx,yy].dot(wgt)
        except IndexError:
            weighted_irf_ph = np.zeros_like(self._irfs_photopeak[...,20,20])
            weighted_irf_nonph_1 = np.zeros_like(self._irfs_nonphoto_1[...,20,20])
            weighted_irf_nonph_2 = np.zeros_like(self._irfs_nonphoto_2[...,20,20])
            
        return weighted_irf_ph, weighted_irf_nonph_1, weighted_irf_nonph_2

    
    def _interpolated_irfs(self, azimuth, zenith):
        """
        Return three lists with the interploated irf curves for
        each detector

        :param azimuth: 
        :param zenith: 
        :returns: 
        :rtype: 

        """

        weighted_irf_ph, weighted_irf_nonph_1, weighted_irf_nonph_2 = self._weighted_irfs(azimuth, zenith)

        
        
        interpolated_irfs_ph = []
        interpolated_irfs_nonph1 = []
        interpolated_irfs_nonph2 = []
        
        for det_number in range(self._n_dets):

            #tmp = interpolate.interp1d(self._energies, weighted_irf[:, det_number])
            
            tmp = log_interp1d(self._energies_database, weighted_irf_ph[:, det_number])
            tmp2 = log_interp1d(self._energies_database, weighted_irf_nonph_1[:, det_number])
            tmp3 = log_interp1d(self._energies_database, weighted_irf_nonph_2[:, det_number])
            
            interpolated_irfs_ph.append(tmp)
            interpolated_irfs_nonph1.append(tmp2)
            interpolated_irfs_nonph2.append(tmp3)
            
        self._current_interpolated_irfs_ph = interpolated_irfs_ph
        self._current_interpolated_irfs_nonph1 = interpolated_irfs_nonph1
        self._current_interpolated_irfs_nonph2 = interpolated_irfs_nonph2

class ResponsePhotopeak(Response):

    def __init__(self, ebounds=None, time=None):
        """
        Init Response object with only Photopeak effective area used
        :param ebound: Ebounds of Ebins
        :param time: Time for which the response should be valid
        :return:
        """
        super(ResponsePhotopeak, self).__init__(ebounds, time)
        
    def _load_irfs(self, time=None):
        """FIXME! briefly describe function
        :param time: Time object, with the time for which the valid response should be used
        :returns: 
        :rtype: 
        """
        
        if time==None:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            print('Using the default irfs. The ones that are valid between 10/05/27 12:45:00'\
                  ' and present (YY/MM/DD HH:MM:SS)')
            
        elif time<Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_0.hdf5')
            print('Using the irfs that are valid between Start'\
                  ' and 03/07/06 06:00:00 (YY/MM/DD HH:MM:SS)')
            
        elif time<Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_1.hdf5')
            print('Using the irfs that are valid between 03/07/06 06:00:00'\
                  ' and 04/07/17 08:20:06 (YY/MM/DD HH:MM:SS)')

        elif time<Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_2.hdf5')
            print('Using the irfs that are valid between 04/07/17 08:20:06'\
                  ' and 09/02/19 09:59:57 (YY/MM/DD HH:MM:SS)')

        elif time<Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
            irf_file = get_path_of_data_file('spi_three_irfs_database_3.hdf5')
            print('Using the irfs that are valid between 09/02/19 09:59:57'\
                  ' and 10/05/27 12:45:00 (YY/MM/DD HH:MM:SS)')

        else:
            irf_file = get_path_of_data_file('spi_three_irfs_database_4.hdf5')
            print('Using the irfs that are valid between 10/05/27 12:45:00'\
                  ' and present (YY/MM/DD HH:MM:SS)')

        irf_database = h5py.File(irf_file, 'r')

        self._energies_database = irf_database['energies'].value

        self._ebounds = self._energies_database
        self._ene_min = self._energies_database[:-1]
        self._ene_max = self._energies_database[1:]
        
        irf_data = irf_database['irfs']

        self._irfs = irf_data[()]

        self._irfs_photopeak = self._irfs[...,0]

        del self._irfs
        
        self._irf_xmin = irf_data.attrs['irf_xmin']
        self._irf_ymin = irf_data.attrs['irf_ymin']
        self._irf_xbin = irf_data.attrs['irf_xbin']
        self._irf_ybin = irf_data.attrs['irf_ybin']
        self._irf_nx = irf_data.attrs['nx']
        self._irf_ny = irf_data.attrs['ny']
        
        irf_database.close()

        self._n_dets = self._irfs_photopeak.shape[1]

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

    def _get_response_det(self, det):
        """
        Get response for a given det
        :param det: Detector ID
        :returns: Full DRM
        """

        interpolated_irfs = self._current_interpolated_irfs_ph[det]
        
        n_energy_bins = len(self._ebounds) - 1
        
        ebins = np.empty((len(self._ene_min), 2))
        eff_area = np.empty_like(ebins)

        ebins[:,0] = self._ene_min
        ebins[:,1] = self._ene_max

        inter = interpolated_irfs(self._ebounds)
        
        eff_area[:,0] = inter[:-1]
        eff_area[:,1] = inter[1:]

        effective_area = integrate.trapz(eff_area, ebins, axis=1)
        
        return effective_area/(self._ene_max-self._ene_min)

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
            weighted_irf_ph = self._irfs_photopeak[..., xx, yy].dot(wgt)

        except IndexError:
            weighted_irf_ph = np.zeros_like(self._irfs_photopeak[...,20,20])

            
        return weighted_irf_ph

    
    def _interpolated_irfs(self, azimuth, zenith):
        """
        Get interploated irf curves for each detector
        :param azimuth: azimuth in sat frame of source
        :param zenith: zenith in sat frame of source
        :returns: 
        """

        weighted_irf_ph = self._weighted_irfs(azimuth, zenith)

        
        
        interpolated_irfs_ph = []
        
        for det_number in range(self._n_dets):

            #tmp = interpolate.interp1d(self._energies, weighted_irf[:, det_number])
            
            tmp = log_interp1d(self._energies_database, weighted_irf_ph[:, det_number])
            
            interpolated_irfs_ph.append(tmp)
            
        self._current_interpolated_irfs_ph = interpolated_irfs_ph

def _prep_out_pixels(ix_left, ix_right, iy_low, iy_up):
    
    left_low = [int(ix_left), int(iy_low)]
    right_low = [int(ix_right), int(iy_low)]
    left_up = [int(ix_left), int(iy_up)]
    right_up = [int(ix_right), int(iy_up)]
    
    out = np.array([left_low, right_low, left_up, right_up]).T

    return out
