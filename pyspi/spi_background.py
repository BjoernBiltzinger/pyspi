import numpy as np
import astropy.io.fits as fits

from pyspi.utils.geometry import cart2polar, polar2cart
from pyspi.io.package_data import get_path_of_data_file

import scipy.interpolate as interpolate
import scipy.integrate as integrate

class SPIBackground(object):

    def __init__(self):
        """
        This class handles the SPI background
        for different analysis cases (GRB, pointsource, diffuse, ...)

        :params: TBD!
        """

        # read in GRB background template for epoch 5 (15 working detectors -- more TBD)
        self._open_spi_grb_background()
        
        
#
#        # construct the misalignment matrix
#        self._open_misalignment_matrix()
#
#
#        # open the space craft pointing file and extract it
#        with fits.open(sc_pointing_file) as f:
#
#            self._pointing_data = f['INTL-ORBI-SCP'].data
#
#
#        # construct the space craft pointings
#
#        self._construct_sc_matrices()



    def _open_spi_grb_background(self):
        """
        Open SPI GRB background file

        :return: None
        """

        # get(???) the path to the data file
        # path to data file
        spi_grb_background_file = 'data/spi_grb_background.fits'

        # open background file
        self._spi_grb_background = fits.open(spi_grb_background_file)

        # define background model energies
        self._emin = self._spi_grb_background['ENERGIES'].data['EMIN']
        self._emax = self._spi_grb_background['ENERGIES'].data['EMAX']
        self._ecen = self._spi_grb_background['ENERGIES'].data['ECEN']
        
        # define background data (patterns as a function of energy)
        self._bg_data_sgl = self._spi_grb_background['SPI.-GRB-BG05'].data['BG_SGL']
        self._bg_data_psd = self._spi_grb_background['SPI.-GRB-BG05'].data['BG_PSD']
        
        # number of detectors
        self._n_dets = self._spi_grb_background['SPI.-GRB-BG05'].data['BG_SGL'].shape[0]


    def interpolated_background_pattern(self):
        """
        Return a list of interpolated detector patterns as a function of energy
        
        :returns: 
        :rtype: 

        """

        interpolated_bg_patterns_sgl = []
        interpolated_bg_patterns_psd = []
        
        
        for det_number in range(self._n_dets):

            tmp_sgl = interpolate.interp1d(self._ecen, self._bg_data_sgl[det_number],fill_value='extrapolate')
            tmp_psd = interpolate.interp1d(self._ecen, self._bg_data_psd[det_number],fill_value='extrapolate')
            
            interpolated_bg_patterns_sgl.append(tmp_sgl)
            interpolated_bg_patterns_psd.append(tmp_psd)
            

        return interpolated_bg_patterns_sgl, interpolated_bg_patterns_psd
    
    
    def get_bg_pattern(self, energies, detectors, event_type):
        """FIXME! briefly describe function

        
        :param energies at which background model for GRBs is read off: 
        :param detector tag:
        :param event_type tag (SGL or PSD):
        :returns: 
        :rtype: 

        """

        interpolated_background_pattern = self.interpolated_background_pattern()

        n_events = len(energies)

        bg_pattern_per_event = []

        for i in range(n_events):
            bg_pattern_per_event = np.append(bg_pattern_per_event,interpolated_background_pattern[event_type[i]][detectors[i]](energies[i]))
            
        return bg_pattern_per_event


    def print_spi_bg(self):
        
        data_sgl = self._spi_grb_background['SPI.-GRB-BG05'].data['BG_SGL']
        data_psd = self._spi_grb_background['SPI.-GRB-BG05'].data['BG_PSD']
        
        return data_sgl, data_psd


                     
    @property
    def bg_data_sgl(self):

        return self._bg_data_sgl
    
    
    @property
    def bg_data_psd(self):

        return self._bg_data_psd
    
    @property
    def bg_energies(self):

        return self._emin, self._emax, self._ecen
     
        
    
    
    
    
