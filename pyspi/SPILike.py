import numpy as np
import pandas as pd
import collections


from astromodels import Parameter
from threeML.plugin_prototype import PluginPrototype
from astropy.io import fits
import matplotlib.pylab as plt
import time

__instrument_name = "INTEGRAL-SPI"


class SPILike(PluginPrototype):


    def __init__(self, name, data, orbit_params):

        # Create the dictionary of nuisance parameters
        # these hold the sky and background fraction
        # for the SPI mask fit.
        # TODO: I have just set them to arbitrary values

        
        self._nuisance_parameters = collections.OrderedDict()

        param_name = "%s_sky" % name

        self._nuisance_parameters[param_name] = Parameter(param_name, 1.0, min_value=0.5, max_value=1.5, delta=0.01)
        self._nuisance_parameters[param_name].fix = True

        param_name = "%s_bkg" % name

        self._nuisance_parameters[param_name] = Parameter(param_name, 1.0, min_value=0.5, max_value=1.5, delta=0.01)
        self._nuisance_parameters[param_name].fix = True

        self._data = data
        self._full_data = data
        self._orbit_params = orbit_params

        super(SPILike, self).__init__(name, self._nuisance_parameters)
    
    @classmethod
    def from_pointing(cls, name, pathname):
        """
        Instance the plugin starting from a oper.fits file.
        :param name: the name for the new instance
        :param filename: path to the file
        :return:
        """    
        # Filenames of files to read
        if pathname[-1] == '/':
            spi_oper_filename = ''.join((pathname, 'spi_oper.fits.gz'))
            sc_orbit_param_filename = ''.join((pathname, 'sc_orbit_param.fits.gz'))
        else:
            spi_oper_filename = '/'.join((pathname, 'spi_oper.fits.gz'))
            sc_orbit_param_filename = '/'.join((pathname, 'sc_orbit_param.fits.gz'))
        
        # Photon counts with energy, time, and event type tags
        with fits.open(spi_oper_filename) as f:
            spi_oper_sgl = f['SPI.-OSGL-ALL'].data
            spi_oper_psd = f['SPI.-OPSD-ALL'].data
            
        # Space craft orbit parameters about orientation, distance etc.
        with fits.open(sc_orbit_param_filename) as f:
            orbit_params = f['INTL-ORBI-SCP'].data
        
        # Combine single events and psd events.
        # Add keyword TYPE (0 = single event, 1 = psd event)            
        sgl_time = spi_oper_sgl['TIME']
        sgl_energy = spi_oper_sgl['ENERGY']
        sgl_detector = spi_oper_sgl['DETE']
        sgl_type = np.zeros(len(sgl_energy)).astype(int)
        
        psd_time = spi_oper_psd['TIME']
        psd_energy = spi_oper_psd['ENERGY']
        psd_detector = spi_oper_psd['DETE']
        psd_type = np.full(len(psd_energy), 1).astype(int)
        
        full_time = np.append(sgl_time, psd_time)
        full_energy = np.append(sgl_energy, psd_energy)
        full_detector = np.append(sgl_detector, psd_detector)
        full_type = np.append(sgl_type, psd_type)
        
        full_data = pd.DataFrame(data={'TIME': full_time,
                                       'ENERGY': full_energy,
                                       'DETECTOR': full_detector,
                                       'TYPE': full_type}, index=None)
        
        # Default energy selection:
        #   - Single events 20 - 400 keV and 1800 - 8000 keV
        #   - PSD events 400 - 1800 keV
        mask_default = ((((full_energy >= 20) & (full_energy < 400)) |
                         ((full_energy > 1800) & (full_energy <= 8000))
                         & (full_type == 0)) |
                        ((full_energy >= 400) & (full_energy <= 1800) &
                         (full_type == 1)))
        
        data = full_data[mask_default]

        return cls(name, data, orbit_params)

    def set_active_measurements(self, energy=None, time=None, event_type=None, *selections):
        """
        Select energy, time, and/or type boundaries
        :param energy: tuple with energy boundaries in keV
        :param time: tuple with time boundaries
        :param event_type: integer for event type (0: single events, 1: psd
        events)
        """
        
        data_energy = np.array(self._full_data['ENERGY'])
        data_time = np.array(self._full_data['TIME'])
        data_type = np.array(self._full_data['TYPE'])
        
        mask_master = np.full(len(self._full_data), True, dtype=bool)
        
        print(mask_master)
        
        if energy:
            print('energy!')
            mask_master = mask_master & ((data_energy >= energy[0]) & (data_energy <= energy[1]))
            print(mask_master)
        if time:
            print('time!')
            mask_master = mask_master & ((data_time >= time[0]) & (data_time <= time[1]))
        if event_type:
            print('type!')
            mask_master = mask_master & (data_type == event_type)
            
        print(mask_master)
        
        self._data = self._full_data[mask_master]
        
    def reset_data(self):
        """
        Reset data block to the original data
        """
        self._data = self._full_data

    def display_detector_mask(self):

        pass

        
    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        pass
    
    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

    

        logL = 0.

        return logL

    def inner_fit(self):

        # here we fix the model parameters
        # and free the mask parameters. TBD.

        logL = self.get_log_like()

        return logL

    def display(self):

        pass
    
    def plot_lightcurve(self):
        
        pass
    
    def plot_spectrum(self):
        
        pass
    
    def plot_detector_pattern(self):
        
        pass
    
    @property
    def data(self):

        return self._data
    
    @property
    def orbit_params(self):

        return self._orbit_params
    
    
def pseudo_detector(detector_hits):
    """
    detector_hits: list of which detectors were hit
    return: pseudo detector id
    """
    # Information from https://heasarc.gsfc.nasa.gov/docs/integral/spi/pages/detectors.html
    hits = np.array(sorted(detector_hits))
    
    double_event_pairs = np.array([(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6),
                          (1, 2), (1, 6), (1, 7), (1, 8), (1, 9), (2, 3),
                          (2, 9), (2, 10), (2, 11), (3, 4), (3, 11), (3, 12),
                          (3, 13), (4, 5), (4, 13), (4, 14), (4, 15), (5, 6),
                          (5, 15), (5, 16), (5, 17), (6, 7), (6, 17), (6, 18),
                          (7, 8), (7, 18), (8, 9), (9, 10), (10, 11), (11, 12),
                          (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
                          (17, 18)])
    
    triple_event_pairs = np.array([(0,1,2), (0,2,3), (0,3,4), (0,4,5),
                          (0,5,6), (0,6,1), (1,2,9), (1,6,7), (1,7,8), (1,8,9),
                          (2,3,11), (2,9,10), (2,10,11), (3,4,13), (3,11,12),
                          (3,12,13), (4,5,15), (4,13,14), (4,14,15), (5,6,17),
                          (5,15,16), (5,16,17), (6,7,18), (6,17,18)])
    
    
    double_det_ids = np.arange(len(double_event_pairs)) + 19
    triple_det_ids = np.arange(len(triple_event_pairs)) + 61
    
    if len(hits) == 2:
        pseudo_detector = double_det_ids[(hits == double_event_pairs).all(axis=1)]
    if len(hits) == 3:
        pseudo_detector = triple_det_ids[(hits == triple_event_pairs).all(axis=1)]
    
    return pseudo_detector

start = time.time()
spilike = SPILike.from_pointing('test', 'data')
print('read', time.time()-start)