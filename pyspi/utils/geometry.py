import numpy as np
from numba import njit

@njit
def polar2cart(ra, dec):
    """
    Convert ra, dec to cartesian
    :param ra: ra coord
    :param dec: dec coord
    :return: cartesian coord vector
    """
    x = np.cos(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    y = np.sin(np.deg2rad(ra)) * np.cos(np.deg2rad(dec))
    z = np.sin(np.deg2rad(dec))

    return np.array([x,y,z])


@njit
def cart2polar(vector):
    """
    Convert cartesian to ra, dec
    :param vector: cartesian coord vector
    :return: ra and dec
    """
    ra = np.arctan2(vector[1],vector[0])
    dec = np.arcsin(vector[2])

    return np.rad2deg(ra), np.rad2deg(dec)
