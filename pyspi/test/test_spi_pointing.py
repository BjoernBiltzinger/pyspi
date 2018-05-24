import numpy as np

import astropy.units as u
from astropy.coordinates import ICRS, Galactic, SkyCoord

from pyspi.spi_pointing import SPIPointing
from pyspi.spi_frame import SPIFrame

from pyspi.io.package_data import get_path_of_data_file

def test_spi_pointing_constructor():

    spi_pointing_file = get_path_of_data_file('sc_orbit_param.fits.gz')

    spi_pointing = SPIPointing(spi_pointing_file)

    # assert that the misalignment matrix has not changed

    default_pointing_matrix = np.array([
    [ 9.99997136e-01,  2.65497922e-04, -2.37837002e-03],
    [-2.69012020e-04,  9.99998873e-01, -1.47732581e-03],
    [ 2.37797511e-03,  1.47796139e-03,  9.99996080e-01]
    ])


    assert  np.allclose(spi_pointing._misalignment_matrix,default_pointing_matrix)

# assert the correct shape of the of all the pointing matrices

    assert spi_pointing.sc_matrix.shape == (417,3,3)

    # assert that the test ra,dec point have not changed

    first_sc_points = spi_pointing.sc_points[0]

    assert first_sc_points['scx_dec'] == 50.62
    assert first_sc_points['scx_ra'] == 206.32

    assert first_sc_points['scy_dec'] == 32.53
    assert first_sc_points['scy_ra'] ==  65.30

    assert first_sc_points['scz_dec'] ==  -19.65926860860834
    assert first_sc_points['scz_ra'] == 142.13



def test_spi_frame():

    spi_pointing_file = get_path_of_data_file('sc_orbit_param.fits.gz')

    spi_pointing = SPIPointing(spi_pointing_file)

    # get the first pointing points
    scp = spi_pointing.sc_points[0]


    # construct a SPI Frame

    spi_frame = SPIFrame(lat=0 * u.deg, lon=0 * u.deg, **scp)

    # test that transformation to ICRS make sense
    icrs_along_x_dir = spi_frame.transform_to(ICRS)

    assert np.allclose(icrs_along_x_dir.ra.value, scp['scx_ra'])

    assert np.allclose(icrs_along_x_dir.dec.value, scp['scx_dec'])

    # assert that angular separation is conserved

    source = SkyCoord(ra=197.075, dec=58.98, unit='deg', frame='icrs')
    center = SkyCoord(ra=206.32522477, dec=50.62928435, unit='deg', frame='icrs')

    icrs_sep = source.separation(center).value

    source_spi = source.transform_to(spi_frame)

    center_spi = center.transform_to(spi_frame)

    spi_sep = spi_frame.separation(source_spi).value


    assert np.allclose(icrs_sep, spi_sep)


    # assert that transforms in and out of frame are conserved



