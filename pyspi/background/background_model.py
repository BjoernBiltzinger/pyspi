import numpy as np
import h5py
from scipy.interpolate import interp1d
from pyspi.utils.livedets import get_live_dets_pointing
from pyspi.io.package_data import get_path_of_internal_data_dir
#from scipy.special import erfc
from math import erfc
from scipy.integrate import quad
from numba import njit, float64
import os

@njit
def trapz(y,x):
    """
    Fast trapz integration with numba
    :param x: x values
    :param y: y values
    :return: Trapz integrated
    """
    return np.trapz(y, x)


@njit
def ln_erfc(x):
    """ 
    logarithmic approximation of the errorfunction
    :param y: value
    :return: log(erfc(y))
    """
    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806,
         0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277]

    t = 1.0 / (1.0 + 0.5*np.abs(x))
    tau = -x*x + (a[0] +
                  t*(a[1] +
                     t*(a[2] +
                        t*(a[3] +
                           t*(a[4] +
                              t*(a[5] +
                                 t*(a[6] +
                                    t*(a[7] +
                                       t*(a[8] +
                                          t*a[9])))))))))

    y = np.log(t) + tau

    if x < 0:
        return np.log(2-np.exp(y))

    return y


@njit
def conv_line_shape(x, amp, mu, sig, tau, ln=True):
    """
    Calculate differential spectrum of a degraded (asymmetric) Gaussian line
    shape
    :param x: Energies where to calculate differential spectrum
    :param amp: Amplitude of line
    :param mu: central energy of line
    :param sig: Width of line
    :param tau: 
    :param ln: Flag
    :return: differential spectrum of a degraded (asymmetric) Gaussian line
    at energies x
    """
    
    if tau < 1e-3:                                  # if tau too small
        val = amp*np.exp(-(x - mu)**2./(2.*sig**2)) # use standard Gaussian
    
    else:
        
        if ln:                                      # logarithmic approximation
            q = np.sqrt(np.pi/2.)*amp*sig/tau
            w = (2.*tau*(x - mu) + sig**2)/(2.*tau**2)  # logarithmic term
            e = ln_erfc((tau*(x - mu) + sig**2)/(np.sqrt(2.)*sig*tau))

            #edx = np.where(np.isfinite(e) == 0)     # set inf values to first
                                                    # non-inf value

            #e[edx] = e[np.where(np.isfinite(e) == 1)[0][0]]
            val = q*np.exp(w + e)
            
    
        else:                                       # standard degraded Gauss
            val = (np.sqrt(np.pi/2.)*amp*sig*np.exp((2.*tau*(x - mu) + sig**2)/
                                                   (2.*tau**2))*
                   (erfc((tau*(x - mu) + sig**2)/
                         (np.sqrt(2.)*sig*tau)))/tau)
        
    return val


@njit
def Powerlaw(x, amp, index, xp):
    """
    Differential spectrum of simple Powerlaw normlized at xp
    :param x: Energies where to calculate differential spectrum
    :param amp: Amplitude of pl
    :param index: Index of pl
    :param xp: Normalization energy
    :return: differential spectrum of a Powerlaw at energies x
    """
    return amp*(x/xp)**index


@njit
def differential_background_spectrum_one_value(E, cont, lines_array,
                                               energy_bounds, average_ebins):
    """
    :return: differential background spectrum for a single energy E
    """
    indices = np.digitize(np.array([E]), bins=energy_bounds)[0]-1
    index = indices
    if index < len(lines_array):
        p = lines_array[index]
        cont_amp, cont_index = cont[index]
        value = Powerlaw(E, cont_amp, cont_index, average_ebins[index])
        for line in range(len(p)):
            line_param = p[line]
            if line_param[0] > 0:
                value += conv_line_shape(E,
                                         line_param[1],
                                         line_param[0],
                                         line_param[2],
                                         line_param[3])
    else:
        value = 0

    return value


@njit
def differential_background_spectrum_array(E, cont, lines_array,
                                           energy_bounds, average_ebins):
    """
    :return: differential background spectrum for an array of energies E
    """
    indices = np.digitize(E, bins=energy_bounds)-1
    values = np.zeros_like(E)
    for i, index in enumerate(indices):
        if index < len(lines_array):
            p = lines_array[index]
            e = E[i]
            # continuum
            cont_amp, cont_index = cont[index]
            values[i] += Powerlaw(e, cont_amp, cont_index,
                                  average_ebins[index])
            # lines
            for line_param in p:
                if line_param[0] > 0:
                    values[i] += conv_line_shape(e,
                                                 line_param[1],
                                                 line_param[0],
                                                 line_param[2],
                                                 line_param[3])
    return values


class BackgroundModelPointing(object):

    def __init__(self, pointing_id):
        """
        Init backgroundmodel caculation for a given pointing
        :param pointing_id: Pointing ID
        :return: Object
        """

        self._dets = get_live_dets_pointing(pointing_id, event_types=["single"])

        self._dets_bkg_model = {}
        for d in self._dets:
            self._dets_bkg_model[d] = BackgroundModelPointingDet(pointing_id, d)

        self.set_tracer(1)

    def set_tracer(self, value):
        """
        Set the tracer value
        :param tracer: tracer value
        :return:
        """
        self._tracer = value

    @property
    def tracer(self):
        """
        :return: tracer value
        """
        return self._tracer

    def bkg_count_rate(self, det, ebounds):
        """
        Get the background count rate for one detector
        :param det: Detector ID
        :param ebounds: Ebounds of Analysis
        :return: background count rates in the ebins
        """
        assert det in self._dets, "Det not valid"

        bkg_det = self._dets_bkg_model[det]

        f = bkg_det.differential_background_spectrum
        bkg_rate = np.zeros(len(ebounds)-1)

        for i, (el, eh) in enumerate(zip(ebounds[:-1], ebounds[1:])):
            bkg_rate[i] += quad(f, el, eh, epsrel=0.01)[0]

        return self.tracer*bkg_det.livetime_fraction*bkg_rate

    def set_livetime(self, det, livetime):
        """
        Set the livetime of one detector
        :param det: Detector ID
        :param livetime: Livetime
        :return:
        """
        self._dets_bkg_model[det].set_livetime(livetime)


class BackgroundModelPointingDet(object):

    def __init__(self, pointing_id, det):
        """
        Init the background model (Siegert et al. 2019) that will be used as
        background estimation. Needed for constant sources.
        For transient sources the polynominal approach is also possible.
        :param pointings_list: List of all pointings that shoud be
        used in the analysis
        :param det: Detector ID
        :return: Object
        """

        self._det = det
        self._pointing_id = pointing_id
        self._rev = int(self._pointing_id[:4])

        self._build_background_spectrum_base()
        #self._set_livetime_fraction()

    def _build_background_spectrum_base(self):
        """
        Get the total base differential background spectrum
        (not normalized to livetime, tracer, etc.)
        :param rev: Revolution
        :param det: Detector
        :return: total differential base spectrum function
        """

        # Get the basis information from hdf5 file
        with h5py.File(os.path.join(
                get_path_of_internal_data_dir(),
                'background_database_new.h5'
        ),
                       'r') as fh5:
            # The bkg model is split in several Ebins which are
            # determined independetly.
            # Read in the boundaries of these Ebins
            self._energy_bounds = np.array(fh5['Ebounds'][()],
                                           dtype=np.float64)

            # Find the correct id for the revolution we want
            revolutions = fh5['Revs'][()]
            i = np.argwhere(revolutions == int(self._rev))[0,0]

            assert i is not None, "Revolution not found in the"\
                "background database..."

        # Average Energy of all Ebins
        self._average_ebins = (self._energy_bounds[1:] +
                               self._energy_bounds[:-1])/2

        # Read in all the line and continuum informations
        # for all background model ebins
        with h5py.File(os.path.join(
                get_path_of_internal_data_dir(),
                'background_database_new.h5'
        ),
                       'r') as fh5:

            self._cont = np.zeros((len(self._energy_bounds)-1, 2))
            all_lines = []

            max_line_num = 0
            for index in range(len(self._energy_bounds)-1):
                ent = fh5["Ebin_{}".format(index)]
                self._cont[index] = ent['Cont'][i, self._det]
                lines = np.array([ent['Line_E'][i, self._det], ent['Line_amp'][i, self._det], ent['Line_width'][i, self._det], ent['Line_tau'][i, self._det]]).T

                all_lines.append(lines)
                if len(ent['Line_E'][i, self._det]) > max_line_num:
                    max_line_num = len(ent['Line_E'][i, self._det])

            self._lines_array = np.ones((len(all_lines), max_line_num, 4))*(-1)
            for i, p in enumerate(all_lines):
                self._lines_array[i, :len(p)] = p

    def differential_background_spectrum(self, E):
        """
        Get differential background spectrum for energy (energies) E
        :param E: Energy as float or Energies as array
        :return: differential background spectrum
        """
        if isinstance(E, (np.ndarray, list)):
            return differential_background_spectrum_array(E,
                                                          self._cont,
                                                          self._lines_array,
                                                          self._energy_bounds,
                                                          self._average_ebins)

        return differential_background_spectrum_one_value(E,
                                                          self._cont,
                                                          self._lines_array,
                                                          self._energy_bounds,
                                                          self._average_ebins)

    def set_livetime(self, livetime):
        """
        Set the livetime
        :param livetime: Livetime value
        :return:
        """
        self._livetime = livetime

    @property
    def livetime(self):
        """
        :return: livetime value
        """
        return self._livetime

    @property
    def livetime_fraction(self):
        """
        :return: livetime fraction
        """
        return 1/(4300*60)
