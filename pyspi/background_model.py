import numpy as np
import h5py
from scipy.integrate import quad, simps
from scipy.interpolate import interp1d
def ln_erfc(x):
    """ 
    logarithmic approximation of the errorfunction
    :param y: value
    :return: log(erfc(y))
    """
    x = np.array(x)

    a = [-1.26551223, 1.00002368, 0.37409196, 0.09678418, -0.18628806, 
         0.27886807, -1.13520398, 1.48851587, -0.82215223, 0.17087277]
      
    t = 1.0 / (1.0 + 0.5*np.abs(x))
    tau = -x*x + (a[0] + t*(a[1] + t*(a[2] + t*(a[3] + t*(a[4] +\
                t*(a[5] + t*(a[6] + t*(a[7] + t*(a[8] + t*a[9])))))))))
    
    y = np.log(t) + tau
    
    lt0 = np.where(x < 0)[0]

    if len(lt0) > 1:

        y[lt0] = np.log(2-np.exp(y[lt0]))

    return y

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
    :return: differential spectrum of a degraded (asymmetric) Gaussian line at energies x
    """
    x = np.array(x)
    
    if tau < 1e-3:                                  # if tau too small
        val = amp*np.exp(-(x - mu)**2./(2.*sig**2)) # use standard Gaussian
    
    else:
        
        if ln:                                      # logarithmic approximation
            q = np.sqrt(np.pi/2.)*amp*sig/tau
            w = (2.*tau*(x - mu) + sig**2)/(2.*tau**2)  # logarithmic term
            e = ln_erfc((tau*(x - mu) + sig**2)/(np.sqrt(2.)*sig*tau))
            edx = np.where(np.isfinite(e) == 0)     # set inf values to first
                                                    # non-inf value
            e[edx] = e[np.where(np.isfinite(e) == 1)[0][0]]
            val = q*np.exp(w + e)
            
    
        else:                                       # standard degraded Gauss
            val = np.sqrt(np.pi/2.)*amp*sig*np.exp((2.*tau*(x - mu) +\
                  sig**2)/(2.*tau**2))*(erfc((tau*(x - mu) +\
                  sig**2)/(np.sqrt(2.)*sig*tau)))/tau
        
    return val

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

class BackgroundModel(object):

    def __init__(self, pointing_list, event_types=["single"], ebounds=None):
        """
        Init the background model (Siegert et al. 2019) that will be used as background estimation.
        Needed for constant sources. For transient sources the polynominal approach is 
        also possible.
        :param pointings_list: List of all pointings that shoud be used in the analysis
        :param event_types: Which event types should be used?
        :param ebounds: Energy boundaries for analysis
        """

        self._event_types = event_types
        self._pointings_list = pointing_list

        # Base energies used in background derivation
        self._energy_base = np.arange(18.25,2000.25,0.5)
        
        # Save given ebounds
        if ebounds is not None:
            self._ebounds = ebounds
            self._ene_min = ebounds[:-1]
            self._ene_max = ebounds[1:]
        else:
            # Default energy bins
            self._ebounds = np.logspace(np.log10(20),np.log10(8000),30)
            self._ene_min = self._ebounds[:-1]
            self._ene_max = self._ebounds[1:]

        #self._build_background_response()

    def _build_background_response(self):
        """
        Build the background reponses for all pointing_ids and needed detectors
        :return:
        """
        bkg_sgl_psd_sum_responses_base = self._build_base_background_response()
        # Norm to livetime of sw
        live_time_sw = self._livetime_sw()
        live_time_orbit = self._livetime_orbit()

        bkg_sgl_psd_sum_responses_livetime_cor = bkg_sgl_psd_sum_responses_base * live_time_sw/live_time_orbit 

        # Norm to whole data

        bkg_sum_not_norm = np.sum(bkg_sgl_psd_sum_responses_livetime_cor, axis=(0,1))
        data_sum = 1000#????

        self._bkg_sgl_psd_sum_response_final = bkg_sgl_psd_sum_responses_livetime_cor*(bkg_sum_not_norm/data_sum)
        
    def _build_base_background_responses(self):
        """
        Build the base background response from the background model. No livetime correction here.
        :return: base background response
        """
        if "single" in self._event_types:
            bkg_sgl_psd_sum_base =  np.zeros((self._pointings_list.size, 19, self._ene_min.size))
            for j, p in enumerate(self._pointings_list):
                rev = p[:4]
                if "single" in self._event_types:
                    for i in range(19):
                        spectrum = self._background_spectrum(rev, i)
                        bkg_rates = spectrum(self._energy_base)
                        inter = interp1d(self._energy_base, bkg_rates)
                        for k, (emin, emax) in enumerate(zip(self._ene_min, self._ene_max)):
                            bkg_sgl_psd_sum_base[j, i, k] = quad(lambda x: inter(x), emin, emax)[0]

        else:
            raise NotImplementedError('Only single event background implemented at the moment!')

        return bkg_sgl_psd_sum_base
    
    def _livetime_orbit(self):
        """
        Get the livetime of the full orbit for every used sw for all dets
        :return:
        """
        if "single" in self._event_types: 
            livetime_orbit_sgl = np.zeros((self._pointings_list.size, 19))
            for j, p in enumerate(self._pointings_list):
                rev = p[:4]
                livetime_orbit = np.zeros(19)
                for i in range(19):
                    livetime_orbit_sgl[j,i] = 1000#??
        else:
            raise NotImplementedError('Only single event background implemented at the moment!')
        
        return livetime_orbit_sgl

    def _livetime_sw(self):
        """
        Get the livetimes of all used sw for all dets
        :return:
        """
        if "single" in self._event_types: 
            livetime_orbit_sgl = np.zeros((self._pointings_list.size, 19))
            for j, p in enumerate(self._pointings_list):
                rev = p[:4]
                livetime_orbit = np.zeros(19)
                for i in range(19):
                    livetime_orbit_sgl[j,i] = 10000#??
        else:
            raise NotImplementedError('Only single event background implemented at the moment!')
        
    
    def _background_spectrum(self, rev, det):
        """
        Get the total base differential background spectrum (not normalized to livetime, tracer, etc.)
        :param rev: Revolution
        :param det: Detector
        :return: total differential base spectrum function
        """

        # Get the basis information from hdf5 file
        with h5py.File('/home/bjorn/Documents/jupyter/pyspi_notebooks/background_database_new.h5', 'r') as fh5:
            # The bkg model is split in several Ebins which are determined independetly.
            # Read in the boundaries of these Ebins
            energy_bounds = fh5['Ebounds'][()]

            # Find the correct id for the revolution we want
            revolutions = fh5['Revs'][()]
            i = np.argwhere(revolutions==int(rev))[0,0]

            assert i is not None, 'Revolution not found in the background database...'

        # Average Energy of all Ebins
        average_ebins = (energy_bounds[1:]+energy_bounds[:-1])/2


        # Read in all the line and continuum informations for all background model ebins
        with h5py.File('/home/bjorn/Documents/jupyter/pyspi_notebooks/background_database_new.h5', 'r') as fh5:
            # Save the params and the errors in two different dicts
            params = {}
            errors = {}
            for index in range(len(energy_bounds)-1):
                ent = fh5["Ebin_{}".format(index)]
                params[index] = {'cont': ent['Cont'][i, det], 'lines': {'energies': ent['Line_E'][i, det], 'amp': ent['Line_amp'][i, det], 'width': ent['Line_width'][i, det], 'tau': ent['Line_tau'][i, det]}}
                errors[index] = {'cont': ent['Cont_err'][i, det], 'lines': {'energies': ent['Line_E_err'][i, det], 'amp': ent['Line_amp_err'][i, det], 'width': ent['Line_width_err'][i, det], 'tau': ent['Line_tau_err'][i, det]}}

        # Build final function
        def differential_background_spectrum(E):
            values=np.zeros_like(E)
            for index in range(len(energy_bounds)-1):
                cont_amp, cont_index = params[index]['cont']
                line_energy, line_amp, line_width, line_tau = params[index]['lines']['energies'], params[index]['lines']['amp'], params[index]['lines']['width'], params[index]['lines']['tau']
                # Continuum
                values = np.where(np.logical_and(E>energy_bounds[index], 
                                                 E<=energy_bounds[index+1]), 
                                  values+Powerlaw(E, cont_amp, cont_index, average_ebins[index]), values)
                # Lines
                for i in range(len(line_energy)):
                    line_val = conv_line_shape(E, line_amp[i], line_energy[i], line_width[i], line_tau[i])
                    values = np.where(np.logical_and(E>energy_bounds[index], 
                                                     E<=energy_bounds[index+1]), 
                                          values+line_val, values)             
            return values
        
        return differential_background_spectrum
