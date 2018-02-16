import numpy as np
import astropy.io.fits as fits

from pyspi.utils.geometry import cart2polar, polar2cart

def _construct_scy(scx_ra, scx_dec, scz_ra, scz_dec):
    
    x = polar2cart(scx_ra, scx_dec)
    z = polar2cart(scz_ra, scz_dec)
    
    return cart2polar(np.cross(x,-z))
    
def _construct_sc_matrix(scx_ra, scx_dec, scy_ra, scy_dec, scz_ra, scz_dec):
    
    sc_matrix = np.zeros((3,3))
    
    sc_matrix[0,:] = polar2cart(scx_ra, scx_dec)
    sc_matrix[1,:] = polar2cart(scy_ra, scy_dec)
    sc_matrix[2,:] = polar2cart(scz_ra, scz_dec)
    
    return sc_matrix




class SPIPointing(object):

    def __init__(self, sc_pointing_file):



        with fits.open(sc_pointing_file) as f:

            self._pointing_data = f['INTL-ORBI-SCP'].data

        self._construct_sc_matrices()


    def _construct_sc_matrices(self):

        self._n_pointings = len(self._pointing_data)

        self._sc_matrix = np.zeros((self._n_pointings,3,3))
        self._sc_points = np.zeros((self._n_pointings, 6))
        
        for i in range(self._n_pointings):

             scx_ra = self._pointing_data['RA_SCX'][time_index]
             scx_dec = self._pointing_data['DEC_SCX'][time_index]
    
             scz_ra = self._pointing_data['RA_SCZ'][time_index]
             scz_dec = self._pointing_data['DEC_SCZ'][time_index]
    
             scy_ra, scy_dec = _construct_scy(scx_ra, scx_dec, scz_ra, scz_dec)
            
             self._sc_matrix[i, ...] = _construct_sc_matrix(scx_ra, scx_dec, scy_ra, scy_dec, scz_ra, scz_dec)
             self._sc_points[i, :] = np.array([scx_ra, scx_dec, scy_ra, scy_dec, scz_ra, scz_dec])
                       
    @property
    def sc_matrix(self):

        return self._sc_matrix

    @property
    def sc_points(self):

        return self._sc_points
        
