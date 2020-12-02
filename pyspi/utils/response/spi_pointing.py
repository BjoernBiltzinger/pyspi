import numpy as np
import astropy.io.fits as fits

from pyspi.utils.geometry import cart2polar, polar2cart
from pyspi.io.package_data import get_path_of_data_file

try:
    from numba import njit
    has_numba = True
except:
    has_numba = False

if has_numba:
    # If numba is available use it to speed up
    @njit
    def _construct_scy(scx_ra, scx_dec, scz_ra, scz_dec):

        x = polar2cart(scx_ra, scx_dec)
        z = polar2cart(scz_ra, scz_dec)

        return cart2polar(np.cross(x,-z))

    @njit
    def _construct_sc_matrix(scx_ra, scx_dec, scy_ra, scy_dec, scz_ra, scz_dec):

        sc_matrix = np.zeros((3,3))

        sc_matrix[0,:] = polar2cart(scx_ra, scx_dec)
        sc_matrix[1,:] = polar2cart(scy_ra, scy_dec)
        sc_matrix[2,:] = polar2cart(scz_ra, scz_dec)

        return sc_matrix

    @njit
    def _transform_icrs_to_spi(ra_icrs, dec_icrs, sc_matrix):
        """
        Calculates lon, lat in spi frame for given ra, dec in ICRS frame and given 
        sc_matrix (sc_matrix pointing dependent)
        :param ra_icrs: Ra in ICRS in degree
        :param dec_icrs: Dec in ICRS in degree
        :param sc_matrix: sc Matrix that gives orientation of SPI in ICRS frame
        :return: lon, lat in spi frame
        """
        # Source in icrs
        vec_ircs = polar2cart(ra_icrs, dec_icrs)
        vec_spi = np.dot(sc_matrix, vec_ircs)
        lon, lat = cart2polar(vec_spi)
        if lon<0:
            lon += 360
        return lon, lat

    @njit
    def _transform_spi_to_icrs(az_spi, zen_spi, sc_matrix):
        """
        Calculates lon, lat in spi frame for given ra, dec in ICRS frame and given
        sc_matrix (sc_matrix pointing dependent)
        :param ra_icrs: Ra in ICRS in degree
        :param dec_icrs: Dec in ICRS in degree
        :param sc_matrix: sc Matrix that gives orientation of SPI in ICRS frame
        :return: lon, lat in spi frame
        """
        # Source in icrs
        vec_spi = polar2cart(az_spi, zen_spi)
        vec_icrs = np.dot(np.linalg.inv(sc_matrix), vec_spi)
        ra, dec = cart2polar(vec_icrs)
        if ra < 0:
            ra += 360
        return ra, dec


else:
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

    def _transform_icrs_to_spi(ra_icrs, dec_icrs, sc_matrix):
        """
        Calculates lon, lat in spi frame for given ra, dec in ICRS frame and given 
        sc_matrix (sc_matrix pointing dependent)
        :param ra_icrs: Ra in ICRS in degree
        :param dec_icrs: Dec in ICRS in degree
        :param sc_matrix: sc Matrix that gives orientation of SPI in ICRS frame
        :return: lon, lat in spi frame
        """
        # Source in icrs
        vec_ircs = polar2cart(ra_icrs, dec_icrs)
        vec_spi = np.dot(sc_matrix, vec_ircs)
        lon, lat = cart2polar(vec_spi)
        if lon<0:
            lon += 360
        return lon, lat

    def _transform_spi_to_icrs(az_spi, zen_spi, sc_matrix):
        """
        Calculates lon, lat in spi frame for given ra, dec in ICRS frame and given
        sc_matrix (sc_matrix pointing dependent)
        :param ra_icrs: Ra in ICRS in degree
        :param dec_icrs: Dec in ICRS in degree
        :param sc_matrix: sc Matrix that gives orientation of SPI in ICRS frame
        :return: lon, lat in spi frame
        """
        # Source in icrs
        vec_spi = polar2cart(az_spi, zen_spi)
        vec_icrs = np.dot(np.linalg.inv(sc_matrix), vec_spi)
        ra, dec = cart2polar(vec_icrs)
        if ra < 0:
            ra += 360
        return ra, dec

class SPIPointing(object):

    def __init__(self, sc_pointing_file):
        """
        This class handles the **current** SPI pointings
        based of the input SPI pointing file

        :param sc_pointing_file: An INTEGRAL/SPI spacecraft pointing file
        """


        # construct the misalignment matrix
        self._open_misalignment_matrix()


        # open the space craft pointing file and extract it
        with fits.open(sc_pointing_file) as f:

            self._pointing_data = f['INTL-ORBI-SCP'].data

        # construct the space craft pointings

        self._construct_sc_matrices()


    def _open_misalignment_matrix(self):
        """
        Open and form the INTEGRAL to SPI misalignment matrix
        that corrects SPI's pointing to the full INTEGRAL
        pointing


        :return: None
        """

        # get the path to the data file
        matrix_file = get_path_of_data_file('inst_misalign_20050328.fits')

        # open the file
        with fits.open(matrix_file) as f:

            # SPI should always be at idx == 2, but lets make sure!
            spi_idx = f['GNRL-IROT-MOD'].data['INSTRUMENT'] == 'SPI'

            # extract the raw matrix
            matrix_raw = f['GNRL-IROT-MOD'].data['MATRIX'][spi_idx]

            # now reshape it

            self._misalignment_matrix = matrix_raw.reshape((3,3))




    def _construct_sc_matrices(self):
        """

        Extract and construct the SPI pointings taking into account the
        misalignment between INTEGRAL and SPI


        :return:
        """

        self._n_pointings = len(self._pointing_data)

        self._sc_matrix = np.zeros((self._n_pointings,3,3))
        self._sc_points = []

        # Look through all times and construct the pointing matrix

        for i in range(self._n_pointings):

            # first wwe construct the matric for INTEGRAL

            scx_ra = self._pointing_data['RA_SCX'][i]
            scx_dec = self._pointing_data['DEC_SCX'][i]

            scz_ra = self._pointing_data['RA_SCZ'][i]
            scz_dec = self._pointing_data['DEC_SCZ'][i]

            scy_ra, scy_dec = _construct_scy(scx_ra, scx_dec, scz_ra, scz_dec)


            intergal_matrix = _construct_sc_matrix(scx_ra, scx_dec, scy_ra, scy_dec, scz_ra, scz_dec)

            # now apply the misalignment matrix

            spi_matrix = np.dot(self._misalignment_matrix, intergal_matrix)

            # now convert the ra and dec to the proper frame

            #scx_ra, scx_dec = np.array([360, 0]) + cart2polar(spi_matrix[0])
            scx_ra, scx_dec = cart2polar(spi_matrix[0])
            scy_ra, scy_dec = cart2polar(spi_matrix[1])
            scz_ra, scz_dec = cart2polar(spi_matrix[2])


            
            self._sc_matrix[i, ...] = spi_matrix
            self._sc_points.append(dict(scx_ra=scx_ra, scx_dec=scx_dec,
                                        scy_ra=scy_ra, scy_dec=scy_dec,
                                        scz_ra=scz_ra, scz_dec=scz_dec))
                       
    @property
    def sc_matrix(self):

        return self._sc_matrix

    @property
    def sc_points(self):

        return self._sc_points
        
