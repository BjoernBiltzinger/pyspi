import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from numba import njit
from astropy.coordinates import (BaseCoordinateFrame,
                                 Attribute,
                                 RepresentationMapping,
                                 frame_transform_graph,
                                 spherical_to_cartesian)

from pyspi.utils.response.spi_pointing import _construct_sc_matrix
from pyspi.utils.geometry import cart2polar, polar2cart

class SPIFrame(BaseCoordinateFrame):
    """
    
    INTEGRAL SPI Frame
    Parameters
    ----------
    representation : `BaseRepresentation` or None
        A representation object or None to have no data (or use the other keywords)
  
    """
    default_representation = coord.SphericalRepresentation

    frame_specific_representation_info = {
        'spherical': [
            RepresentationMapping(
                reprname='lon', framename='lon', defaultunit=u.degree),
            RepresentationMapping(
                reprname='lat', framename='lat', defaultunit=u.degree),
            RepresentationMapping(
                reprname='distance', framename='DIST', defaultunit=None)
        ],
        'unitspherical': [
            RepresentationMapping(
                reprname='lon', framename='lon', defaultunit=u.degree),
            RepresentationMapping(
                reprname='lat', framename='lat', defaultunit=u.degree)
        ],
        'cartesian': [
            RepresentationMapping(
                reprname='x', framename='SCX'), RepresentationMapping(
                reprname='y', framename='SCY'), RepresentationMapping(
                reprname='z', framename='SCZ')
        ]
    }

    # Specify frame attributes required to fully specify the frame
    scx_ra = Attribute(default=None)
    scx_dec = Attribute(default=None)
    
    scy_ra = Attribute(default=None)
    scy_dec = Attribute(default=None)
    
    scz_ra = Attribute(default=None)
    scz_dec = Attribute(default=None)
   




@frame_transform_graph.transform(coord.FunctionTransform, SPIFrame, coord.ICRS)
def spi_to_j2000(spi_coord, j2000_frame):
    """ 
    Transform spi fram to ICRS frame
    """

    sc_matrix = _construct_sc_matrix(spi_coord.scx_ra,
                                    spi_coord.scx_dec,
                                    spi_coord.scy_ra,
                                    spi_coord.scy_dec,
                                    spi_coord.scz_ra,
                                    spi_coord.scz_dec)

    # X,Y,Z = gbm_coord.cartesian

    pos = spi_coord.cartesian.xyz.value

    X0 = np.dot(sc_matrix[:, 0], pos)
    X1 = np.dot(sc_matrix[:, 1], pos)
    X2 = np.clip(np.dot(sc_matrix[:, 2], pos), -1., 1.)

    dec = np.pi / 2. - np.arccos(X2)# np.arcsin(X2)

    idx = np.logical_and(np.abs(X0) < 1E-6, np.abs(X1) < 1E-6)

    ra = np.zeros_like(dec)

    ra[~idx] = np.arctan2(X1[~idx], X0[~idx]) % (2 * np.pi)

    return coord.ICRS(ra=ra * u.radian, dec=dec * u.radian)


@frame_transform_graph.transform(coord.FunctionTransform, coord.ICRS, SPIFrame)
def j2000_to_spi(j2000_frame, spi_coord):
    """ 
    Transform icrs frame to SPI frame
    """

    sc_matrix = _construct_sc_matrix(spi_coord.scx_ra,
                                    spi_coord.scx_dec,
                                    spi_coord.scy_ra,
                                    spi_coord.scy_dec,
                                    spi_coord.scz_ra,
                                    spi_coord.scz_dec)
  

    pos = j2000_frame.cartesian.xyz.value

    X0 = np.dot(sc_matrix[0, :], pos)
    X1 = np.dot(sc_matrix[1, :], pos)
    X2 = np.dot(sc_matrix[2, :], pos)

    lat = np.pi / 2. - np.arccos(X2)  # convert to proper frame

    idx = np.logical_and(np.abs(X0) < 1E-6, np.abs(X1) < 1E-6)

    lon = np.zeros_like(lat)

    lon[~idx] = np.arctan2(X1[~idx], X0[~idx]) % (2 * np.pi)

    return SPIFrame(
        lon=lon * u.radian,
        lat=lat * u.radian,
        scx_ra=spi_coord.scx_ra,
        scx_dec=spi_coord.scx_dec,
        scy_ra=spi_coord.scy_ra,
        scy_dec=spi_coord.scy_dec,
        scz_ra=spi_coord.scz_ra,
        scz_dec=spi_coord.scz_dec

    )

# Functions to do the coordinate transformation fast! But has none
# of the astropy benefits.
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
    if lon < 0:
        lon += 360
    return lon, lat

@njit
def _transform_spi_to_icrs(az_spi, zen_spi, sc_matrix):
    """
    Calculates lon, lat in spi frame for given ra, dec in ICRS frame and given
    sc_matrix (sc_matrix pointing dependent)
    :param az_spi: azimuth in SPI coord system in degree
    :param zen_spi: zenit in SPI coord system in degree
    :param sc_matrix: sc Matrix that gives orientation of SPI in ICRS frame
    :return: ra, dex in ICRS in deg
    """
    # Source in icrs
    vec_spi = polar2cart(az_spi, zen_spi)
    vec_icrs = np.dot(np.linalg.inv(sc_matrix), vec_spi)
    ra, dec = cart2polar(vec_icrs)
    if ra < 0:
        ra += 360
    return ra, dec
