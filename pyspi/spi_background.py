import numpy as np
import astropy.io.fits as fits

from pyspi.utils.geometry import cart2polar, polar2cart
from pyspi.io.package_data import get_path_of_data_file


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
        spi_grb_background_file = 'data/spi_grb_background.fits.gz'

        # open background file
        self._spi_grb_background = fits.open(spi_grb_background_file)

        self._bg_data = self._spi_grb_background['SPI.-BMOD-DSP'].data['COUNTS']



    def print_spi_bg(self):
        
        data = self._spi_grb_background['SPI.-BMOD-DSP'].data['COUNTS']
        
        return data


                     
    @property
    def bg_data(self):

        return self._bg_data
    
    
    
    
    
    
