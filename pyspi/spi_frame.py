import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, Attribute, RepresentationMapping
from astropy.coordinates import frame_transform_graph, spherical_to_cartesian
    
from pyspi.spi_pointing import _construct_sc_matrix


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
