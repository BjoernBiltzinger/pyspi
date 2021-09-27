import numpy as np
import astropy.io.fits as fits
from numba import njit

from pyspi.utils.geometry import cart2polar, polar2cart
from pyspi.io.package_data import get_path_of_data_file

@njit
def _construct_scy(scx_ra, scx_dec, scz_ra, scz_dec):
    """
    Construct the vector of the y-axis of the Integral coord system
    in the ICRS frame
    :pararm scx_ra: ra coordinate of satellite x-axis in ICRS
    :pararm scx_dec: dec coordinate of satellite x-axis in ICRS
    :pararm scz_ra: ra coordinate of satellite z-axis in ICRS
    :pararm scz_dec: dec coordinate of satellite z-axis in ICRS
    :return: vector of the y-axis of the Integral coord system
    in the ICRS frame
    """
    x = polar2cart(scx_ra, scx_dec)
    z = polar2cart(scz_ra, scz_dec)

    return cart2polar(np.cross(x, -z))

@njit
def _construct_sc_matrix(scx_ra, scx_dec, scy_ra, scy_dec, scz_ra, scz_dec):
    """
    Construct the sc_matrix, with which we can transform ICRS <-> Sat. Frame
    :pararm scx_ra: ra coordinate of satellite x-axis in ICRS
    :pararm scx_dec: dec coordinate of satellite x-axis in ICRS
    :pararm scy_ra: ra coordinate of satellite y-axis in ICRS
    :pararm scy_dec: dec coordinate of satellite y-axis in ICRS
    :pararm scz_ra: ra coordinate of satellite z-axis in ICRS
    :pararm scz_dec: dec coordinate of satellite z-axis in ICRS
    :return: sc_matrix (3x3)
    """

    sc_matrix = np.zeros((3,3))

    sc_matrix[0, :] = polar2cart(scx_ra, scx_dec)
    sc_matrix[1, :] = polar2cart(scy_ra, scy_dec)
    sc_matrix[2, :] = polar2cart(scz_ra, scz_dec)

    return sc_matrix


class SPIPointing(object):

    def __init__(self, sc_pointing_file):
        """
        This class handles the **current** SPI pointings
        based of the input SPI pointing file

        :param sc_pointing_file: An INTEGRAL/SPI spacecraft pointing file
        :return:
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
        Open and build the INTEGRAL to SPI misalignment matrix
        that corrects SPI's pointing to the full INTEGRAL
        pointing
        :return:
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

            intergal_matrix = _construct_sc_matrix(scx_ra, scx_dec, scy_ra,
                                                   scy_dec, scz_ra, scz_dec)

            # now apply the misalignment matrix
            spi_matrix = np.dot(self._misalignment_matrix, intergal_matrix)

            # now convert the ra and dec to the proper frame
            scx_ra, scx_dec = cart2polar(spi_matrix[0])
            scy_ra, scy_dec = cart2polar(spi_matrix[1])
            scz_ra, scz_dec = cart2polar(spi_matrix[2])

            self._sc_matrix[i, ...] = spi_matrix
            self._sc_points.append(dict(scx_ra=scx_ra, scx_dec=scx_dec,
                                        scy_ra=scy_ra, scy_dec=scy_dec,
                                        scz_ra=scz_ra, scz_dec=scz_dec))

    @property
    def sc_matrix(self):
        """
        :return: sc_matrix of all the pointings
        """
        return self._sc_matrix

    @property
    def sc_points(self):
        """
        :return: ra, dec coordinates of the SPI x,y and z axis in the
        ICRS frame for all the pointings
        """
        return self._sc_points
        
