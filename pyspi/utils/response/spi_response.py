import os
from datetime import datetime
import numpy as np
import scipy.interpolate as interpolate
from astropy.time.core import Time
from IPython.display import HTML
from numba import njit
from interpolation import interp
import copy

from pyspi.io.get_files import get_files
from pyspi.io.package_data import (get_path_of_data_file,
                                   get_path_of_external_data_dir)
from pyspi.utils.response.spi_pointing import SPIPointing
from pyspi.utils.response.spi_frame import (_transform_icrs_to_spi,
                                            _transform_spi_to_icrs)
from pyspi.utils.response.spi_response_data import (ResponseDataPhotopeak,
                                                    ResponseDataRMF)
from pyspi.utils.function_utils import find_needed_ids


@njit
def trapz(y, x):
    """
    Fast trapz integration with numba
    :param x: x values
    :param y: y values
    :return: Trapz integrated
    """
    return np.trapz(y, x)


@njit
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

    return np.power(10., lin_interp)


@njit
def add_frac(ph_matrix, i, idx, ebounds, einlow, einhigh):
    """
    Recursive Funktion to get the fraction of einlow...
    """
    if idx+1 == len(ebounds):
        pass
    elif ebounds[idx+1] >= einhigh:
        ph_matrix[i, idx] =\
            np.min(
                np.array(
                    [1,
                     (einhigh-ebounds[idx])/
                     (einhigh-einlow)]
                )
            )

    else:
        frac = np.min(np.array([(ebounds[idx+1]-einlow)/(einhigh-einlow),
                               (ebounds[idx+1]-ebounds[idx])/(einhigh-einlow)]))
        ph_matrix[i, idx] = frac

        add_frac(ph_matrix, i, idx+1, ebounds, einlow, einhigh)


@njit(fastmath=True)
def _get_xy_pos(azimuth, zenith, xmin, ymin, xbin, ybin):
    """
    Get the xy position on the reponse grid for given azimuth and zenith
    :param azimuth: Azmiuth angle in rad
    :param zenith: Zenith angle in rad
    :param xmin: Smallest x-grid entry
    :param ymin: Smallest y-grid entry
    :param xbin: Size of bins in x-direction
    :param ybin: Size of bins in y-direction
    :return: Grid postition (x,y)
    """

    x = np.cos(azimuth)*np.cos(zenith)
    y = np.sin(azimuth)*np.cos(zenith)
    z = np.sin(zenith)

    zenith_pointing = np.arccos(x)
    azimuth_pointing = np.arctan2(z,y)

    x_pos = (zenith_pointing * np.cos(azimuth_pointing) - xmin) / xbin
    y_pos = (zenith_pointing * np.sin(azimuth_pointing) - ymin) / ybin

    return x_pos, y_pos


def _prep_out_pixels(ix_left, ix_right, iy_low, iy_up):
    """
    Simple function to get the 2D-indices of the 4 points defined by
    given pairs of indices in x and y direction
    :param ix_left: x-index of the left points
    :param ix_right: x-index of the right points
    :param iy_low: y-index of the bottom points
    :param iy_up: y-index of the top points
    :return: array with the 4 2D indices defining the 4 grid points
    """

    left_low = [int(ix_left), int(iy_low)]
    right_low = [int(ix_right), int(iy_low)]
    left_up = [int(ix_left), int(iy_up)]
    right_up = [int(ix_right), int(iy_up)]

    out = np.array([left_low, right_low, left_up, right_up]).T

    return out


def multi_response_irf_read_objects(times, detector, drm='Photopeak'):
    """
    TODO: This is very ugly. Come up with a better way.
    Function to initalize the needed responses for the given times.
    Only initalize every needed response version once! Because of memory.
    One response object needs about 1 GB of RAM...
    TODO: Not needed at the moment. We need this when we want to analyse
    many pointings together.
    :param times: Times of the different sw used
    :return: list with correct response version object of the times
    """
    response_versions = []
    for time in times:
        if not time:
            # Default latest response version
            response_versions.append(4)

        elif time < Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
            response_versions.append(0)

        elif time < Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
            response_versions.append(1)

        elif time < Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
            response_versions.append(2)

        elif time < Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
            response_versions.append(3)

        else:
            response_versions.append(4)

    responses = [None, None, None, None, None]

    response_irf_read_times = []
    for version in response_versions:
        if responses[version] is None:
            # Create this response object
            if drm == "Photopeak":
                responses[version] = ResponseDataPhotopeak(
                    detector=detector,
                    version=version)
            else:
                responses[version] = ResponseDataRMF(version=version)
                
        response_irf_read_times.append(responses[version])
    return response_irf_read_times


class ResponseGenerator(object):
    def __init__(self,
                 pointing_id=None,
                 ebounds=None,
                 response_irf_read_object=None,
                 det=None):
        """
        Base Response Class - Here we have everything that stays the same for
        GRB and Constant Pointsource Reponses
        :param ebounds: User defined ebins for binned effective area
        :param response_irf_read_object: Object that holds
        the read in irf values
        :param sc_matrix: Matrix to convert SPI coordinate system <-> ICRS
        :param det: Which detector
        :returns: Object
        """
        # Get the data, either from afs or from ISDC archive
        try:
            # Get the data from the afs server
            get_files(pointing_id, access="afs")
        except AssertionError:
            # Get the files from the iSDC data archive
            print("AFS data access did not work. "
                  "I will try the ISDC data archive.")
            get_files(pointing_id, access="isdc")

        # Read in geometry file to get sc_matrix
        geometry_file_path = os.path.join(get_path_of_external_data_dir(),
                                          'pointing_data',
                                          pointing_id,
                                          'sc_orbit_param.fits.gz')

        pointing_object = SPIPointing(geometry_file_path)
        sc_matrix = pointing_object.sc_matrix[10]

        self._irf_ob = response_irf_read_object
        self._ebounds = ebounds
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)
        self._sc_matrix = sc_matrix
        self._det = det
        self._pointing_id = pointing_id

    def set_binned_data_energy_bounds(self, ebounds):
        """
        Change the energy bins for the binned effective_area
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins,
        ebounds[1:] end of ebins
        :return:
        """

        # if the new bins are not the old ones: update them
        if not np.array_equal(ebounds, self._ebounds):
            self._ene_min = ebounds[:-1]
            self._ene_max = ebounds[1:]
            self._ebounds = ebounds
    
    def get_xy_pos(self, azimuth, zenith):
        """
        Get xy position (in SPI simulation) for given azimuth and zenith
        :param azimuth: Azimuth in Sat. coordinates [rad]
        :param zenith: Zenith in Sat. coordinates [rad]
        :returns: grid position in (x,y) coordinates
        """
        # we call a numba function here to speed it up
        return _get_xy_pos(azimuth,
                           zenith,
                           self.irf_ob.irf_xmin,
                           self.irf_ob.irf_ymin,
                           self.irf_ob.irf_xbin,
                           self.irf_ob.irf_ybin)
    
    def set_location(self, ra, dec):
        """
        Calculate the weighted irfs for the three
        event types for a given position
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

    def set_location_direct_sat_coord(self, azimuth, zenith):
        """
        Calculate the weighted irfs for the three
        event types for a given position
        :param azimuth: Azimuth position in sat frame
        :param zenith: Zenith position in sat frame
        :returns: ra and dec value
        """

        self._weighted_irfs(np.deg2rad(azimuth),
                            np.deg2rad(zenith))

        self._recalculate_response()

        return _transform_spi_to_icrs(azimuth,
                                      zenith,
                                      self._sc_matrix)

    def _weighted_irfs(self, azimuth, zenith):
        """
        Calculate the weighted irfs for the three event
        types for a given position
        :param azimuth: Azimuth position in sat frame
        :param zenith: Zenith position in sat frame
        :returns:
        """
        raise NotImplementedError("Must be implemented in child class.")

    def _recalculate_response(self):

        raise NotImplementedError("Must be implemented in child class.")

    def _get_irf_weights(self, x_pos, y_pos):
        """
        Get the 4 grid points around (x_pos, y_pos) and the weights
        for a rectangular interpolation
        :param x_pos: grid position x-coordinates
        :param y_pos: grid position y-coordinates
        :returns: weights, x-indices, y-indices of the 4 closest points
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

        # Get the weights
        if ix_left < 0.:

            if ix_right < 0.:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

            ix_left = ix_right
            wgt_left = 0.5
            wgt_right = 0.5

        elif ix_right >= self.irf_ob.irf_nx:

            if ix_left >= self.irf_ob.irf_nx:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

            ix_right = ix_left
            wgt_left = 0.5
            wgt_right = 0.5

        else:

            wgt_left = 1. - wgt_right

        if iy_low < 0:
            if iy_up < 0:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

            iy_low = iy_up
            wgt_up = 0.5
            wgt_low = 0.5

        elif iy_up >= self.irf_ob.irf_ny:

            if iy_low >= self.irf_ob.irf_ny:

                out = _prep_out_pixels(ix_left, ix_right, iy_low, iy_up)

                return wgt, out[0], out[1]

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

    @property
    def irf_ob(self):
        """
        :return: the irf_read object with the information from the response
        simulation
        """
        return self._irf_ob

    @property
    def det(self):
        """
        :return: detector number
        """
        return self._det

    @property
    def ebounds(self):
        """
        :return: Ebounds of the analysis
        """
        return self._ebounds

    @property
    def ene_min(self):
        """
        :return: Start of ebounds
        """
        return self._ene_min

    @property
    def ene_max(self):
        """
        :return: End of Ebounds
        """
        return self._ene_max

    @property
    def rod(self):
        """
        Ensure that you know what you are doing.

        :return: Roland
        """
        return HTML(filename=get_path_of_data_file('roland.html'))

class ResponseRMFGenerator(ResponseGenerator):

    def __init__(self,
                 pointing_id=None,
                 monte_carlo_energies=None,
                 ebounds=None,
                 response_irf_read_object=None,
                 det=None,
                 fixed_rsp_matrix=None):
        """
        Init Response object with total RMF used
        :param pointing_id: The pointing ID for which the
        response should be valid
        :param ebound: Ebounds of Ebins
        :param monte_carlo_energies: Input energy bin edges
        :param response_irf_read_object: Object that holds
        the read in irf values
        :param det: Detector ID
        :param fixed_rsp_matrix: A fixed response matrix to overload
        the normal matrix
        :return: Object
        """
        assert isinstance(response_irf_read_object, ResponseDataRMF)

        super(ResponseRMFGenerator, self).__init__(
            pointing_id=pointing_id,
            ebounds=ebounds,
            response_irf_read_object=response_irf_read_object,
            det=det)

        self._monte_carlo_energies = monte_carlo_energies

        if fixed_rsp_matrix is None:
            self._rebin_rmfs()
            self._given_rsp_mat = False
            self._rsp_matrix = None
        else:
            self._given_rsp_mat = True
            self._rsp_matrix = fixed_rsp_matrix

        self._weighted_irf_ph = None
        self._weighted_irf_nonph_1 = None
        self._weighted_irf_nonph_2 = None

    @classmethod
    def from_time(cls,
                  time,
                  det,
                  ebounds,
                  monte_carlo_energies,
                  rsp_read_obj,
                  fixed_rsp_matrix=None):
        """
        Init Response object with total RMF used from a time
        :param time: Time for which to construct the response object
        :param ebound: Ebounds of Ebins
        :param monte_carlo_energies: Input energy bin edges
        :param response_irf_read_object: Object that holds
        the read in irf values
        :param det: Detector ID
        :param fixed_rsp_matrix: A fixed response matrix to overload
        the normal matrix
        :return: Object
        """

        pointing_id = find_needed_ids(time)

        return cls(
            pointing_id=pointing_id,
            monte_carlo_energies=monte_carlo_energies,
            ebounds=ebounds,
            response_irf_read_object=rsp_read_obj,
            det=det,
            fixed_rsp_matrix=fixed_rsp_matrix
        )

    def _rebin_rmfs(self):
        """
        Rebin the base rmf shape matrices for the given ebounds and
        incoming energies
        :return:
        """
        # Number of ebins on input and output side
        N_ebins = len(self.ebounds)-1
        N_monte_carlo = len(self.monte_carlo_energies)

        # get the interpolation grid (ein and the log mean of the eout_bins)
        log_eout_mean = 10**((np.log10(self.ebounds[:-1]) +
                              np.log10(self.ebounds[1:])) / 2.0)
        eout_width = self.ebounds[1:] - self.ebounds[:-1]
        xx, yy = np.meshgrid(self.monte_carlo_energies,
                             log_eout_mean)
        points = np.array((xx.ravel(), yy.ravel())).T

        # build the interpolation functions for rmf_2 mat and rmf_3 mat
        base_ebounds = self.irf_ob.ebounds_rmf_2_base
        base_ebounds_width = base_ebounds[1:]-base_ebounds[:-1]

        x_base = self.irf_ob.energies_database
        y_base = 10**((np.log10(base_ebounds[:-1])+
                       np.log10(base_ebounds[1:]))/2.0)

        lin_int_rmf2 =\
            interpolate.RegularGridInterpolator((x_base, y_base),
                                                self.irf_ob.rmf_2_base /
                                                base_ebounds_width,
                                                bounds_error=False,
                                                fill_value=0)

        lin_int_rmf3 =\
            interpolate.RegularGridInterpolator((x_base, y_base),
                                                self.irf_ob.rmf_3_base /
                                                base_ebounds_width,
                                                bounds_error=False,
                                                fill_value=0)

        # call the interpolation and create the correct matrix shape
        mat2inter = (lin_int_rmf2(points).reshape(N_ebins, N_monte_carlo).T)

        mat3inter = (lin_int_rmf3(points).reshape(N_ebins, N_monte_carlo).T)

        # trapz integrate over out bins

        self._mat2inter = 0.5*(mat2inter[1:]+mat2inter[:-1])*eout_width
        self._mat3inter = 0.5*(mat3inter[1:]+mat3inter[:-1])*eout_width

        # normalize the rows

        norm_fact = np.sum(self._mat2inter, axis=1)

        # Normalize this - is this correct?
        self._mat2inter[norm_fact>0] /= norm_fact[norm_fact>0][:, np.newaxis]
        self._mat2inter[norm_fact<=0] = 0

        norm_fact = np.sum(self._mat3inter, axis=1)

        # Normalize this - is this correct?
        self._mat3inter[norm_fact>0] /= norm_fact[norm_fact>0][:, np.newaxis]
        self._mat3inter[norm_fact<=0] = 0

        self._ph_matrix = np.zeros((N_monte_carlo-1,
                                    N_ebins))

        for i, (einlow, einhigh) in \
            enumerate(zip(self.monte_carlo_energies[:-1],
                          self.monte_carlo_energies[1:])):
            if (not einhigh < self.ebounds[0]) and (not einlow > self.ebounds[-1]):
                # find id and fraction
                idx = np.argwhere(self.ebounds > einlow)[0, 0]-1
                if idx==-1:
                    idx=0
                add_frac(self._ph_matrix, i, idx, self.ebounds, einlow, einhigh)

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
            self._weighted_irf_ph = \
                self.irf_ob.irfs_photopeak[:, self._det, xx, yy].dot(wgt)
            self._weighted_irf_nonph_1 = \
                self.irf_ob.irfs_nonphoto_1[:, self._det, xx, yy].dot(wgt)
            self._weighted_irf_nonph_2 = \
                self.irf_ob.irfs_nonphoto_2[:, self._det, xx, yy].dot(wgt)
        except IndexError:
            self._weighted_irf_ph =\
                np.zeros_like(self.irf_ob.irfs_photopeak[:, self._det, 20, 20])
            self._weighted_irf_nonph_1\
                = np.zeros_like(self.irf_ob.irfs_nonphoto_1[:, self._det, 20, 20])
            self._weighted_irf_nonph_2\
                = np.zeros_like(self.irf_ob.irfs_nonphoto_2[:, self._det, 20, 20])

    def _recalculate_response(self):
        """
        Get response for the current position
        :returns:
        """
        if self._given_rsp_mat:
            self._matrix = self._rsp_matrix
        else:
            ein_bins = np.empty((len(self._monte_carlo_energies)-1, 2))

            ein_bins[:, 0] = self._monte_carlo_energies[:-1]
            ein_bins[:, 1] = self._monte_carlo_energies[1:]

            monte_carlo_log_mean = 10**((
                np.log10(self._monte_carlo_energies[1:]) +
                np.log10(self._monte_carlo_energies[:-1]))/2.0)

            interph = log_interp1d(monte_carlo_log_mean,
                                   self.irf_ob.energies_database,
                                   self._weighted_irf_ph)
            inter2 = log_interp1d(monte_carlo_log_mean,
                                  self.irf_ob.energies_database,
                                  self._weighted_irf_nonph_1)
            inter3 = log_interp1d(monte_carlo_log_mean,
                                  self.irf_ob.energies_database,
                                  self._weighted_irf_nonph_2)

            # TODO clean up these .T calls and maybe change
            # the trapz to logmean

            mat1 = (inter2*self._mat2inter.T).T
            mat2 = (inter3*self._mat3inter.T).T

            # Add photopeak to the ebin with the photon energy within
            # its bounds
            self._transpose_matrix = mat1+mat2+(interph*self._ph_matrix.T).T

            self._matrix = self._transpose_matrix.T

    def clone(self):
        """
        Clone this response object
        :return: cloned response
        """
        return ResponseRMFGenerator(
            pointing_id=copy.deepcopy(self._pointing_id),
            monte_carlo_energies=copy.deepcopy(self.monte_carlo_energies),
            ebounds=copy.deepcopy(self.ebounds),
            response_irf_read_object=self.irf_ob,
            det=copy.deepcopy(self.det),
            fixed_rsp_matrix=copy.deepcopy(self._rsp_matrix)
        )

    @property
    def matrix(self):
        """
        :return: response matrix
        """
        return self._matrix

    @property
    def transpose_matrix(self):
        """
        :return: transposed response matrix
        """
        return self._transpose_matrix

    @property
    def monte_carlo_energies(self):
        """
        :return: Input energies for response
        """
        return self._monte_carlo_energies


class ResponsePhotopeakGenerator(ResponseGenerator):

    def __init__(self,
                 pointing_id=None,
                 ebounds=None,
                 response_irf_read_object=None,
                 det=None):
        """
        Init Response object with photopeak only
        :param pointing_id: The pointing ID for which the
        response should be valid
        :param ebound: Ebounds of Ebins
        :param response_irf_read_object: Object that holds
        the read in irf values
        :param det: Detector ID
        :return: Object
        """
        assert isinstance(response_irf_read_object, ResponseDataPhotopeak)

        # call init of base class
        super(ResponsePhotopeakGenerator, self).__init__(
            pointing_id,
            ebounds,
            response_irf_read_object,
            det)

        self._effective_area = None
        self._weighted_irf_ph = None

    def _weighted_irfs(self, azimuth, zenith):
        """
        Calculate the weighted irfs for the three event types for a given
        position
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
            self._weighted_irf_ph = \
                self.irf_ob.irfs_photopeak[:,self._det, xx, yy].dot(wgt)

        except IndexError:
            self._weighted_irf_ph =\
                np.zeros_like(self.irf_ob.irfs_photopeak[:, self._det, 20, 20])

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
        
        inter = log_interp1d(self.ebounds,
                             self.irf_ob.energies_database,
                             self._weighted_irf_ph)
        
        eff_area[:, 0] = inter[:-1]
        eff_area[:, 1] = inter[1:]
        
        self._effective_area = trapz(eff_area, ebins)/(self._ene_max -
                                                       self._ene_min)

    @classmethod
    def from_time(cls,
                  time,
                  det,
                  ebounds,
                  rsp_read_obj,):
        """
        Init Response object with photopeak only
        :param time: The time for which the
        response should be valid
        :param ebound: Ebounds of Ebins
        :param response_irf_read_object: Object that holds
        the read in irf values
        :param det: Detector ID
        :return: Object
        """
        pointing_id = find_needed_ids(time)

        return cls(
            pointing_id=pointing_id,
            ebounds=ebounds,
            response_irf_read_object=rsp_read_obj,
            det=det,
        )

    @property
    def effective_area(self):
        """
        :return: vector with photopeak effective area
        """
        return self._effective_area
