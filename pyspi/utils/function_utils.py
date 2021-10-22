import numpy as np
from astropy.time.core import Time, TimeDelta
from datetime import datetime
import h5py

from pyspi.io.package_data import get_path_of_data_file


def get_time_object(time):
    """
    Transform the input into a time object. Input can either be a
    time object or a string with the format "YYMMDD HHMMSS"
    :param time: time object or a string with the format "YYMMDD HHMMSS"
    :return: time object
    """
    if not isinstance(time, Time):
        time = datetime.strptime(time,
                                 '%y%m%d %H%M%S')
        time = Time(time)
    return time


def leapseconds(time_object):
    """
    Hard coded leap seconds from start of INTEGRAL to time of time_object
    :param time_object: Time object to which the number of
    leapseconds should be detemined
    :return: TimeDelta object of the needed leap seconds
    """

    time_object = get_time_object(time_object)

    if time_object < Time(datetime.strptime('060101 000000',
                                            '%y%m%d %H%M%S')):
        lsec = 0
    elif time_object < Time(datetime.strptime('090101 000000',
                                              '%y%m%d %H%M%S')):
        lsec = 1
    elif time_object < Time(datetime.strptime('120701 000000',
                                              '%y%m%d %H%M%S')):
        lsec = 2
    elif time_object < Time(datetime.strptime('150701 000000',
                                              '%y%m%d %H%M%S')):
        lsec = 3
    elif time_object < Time(datetime.strptime('170101 000000',
                                              '%y%m%d %H%M%S')):
        lsec = 4
    else:
        lsec = 5
    return TimeDelta(lsec, format='sec')


def find_response_version(time):
    """
    Find the correct response version number for a given time
    :param time: time of interest
    :return: response version number
    """

    time = get_time_object(time)

    if time < Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
        version = 0
    elif time < Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
        version = 1
    elif time < Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
        version = 2
    elif time < Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
        version = 3
    else:
        version = 4
    return version

def find_needed_ids(time):
    """
    Get the pointing id of the needed data to cover the GRB time
    :return: Needed pointing id
    """

    time = get_time_object(time)

    # Path to file, which contains id information and start and stop
    # time
    id_file_path = get_path_of_data_file('id_data_time.hdf5')

    # Get GRB time in ISDC_MJD
    time_of_GRB_ISDC_MJD = (time).tt.mjd-51544
    # Get which id contain the needed time. When the wanted time is
    # too close to the boundarie also add the pervious or following
    # observation id
    id_file = h5py.File(id_file_path, 'r')
    start_id = id_file['Start'][()]
    stop_id = id_file['Stop'][()]
    ids = id_file['ID'][()]

    mask_larger = start_id < time_of_GRB_ISDC_MJD
    mask_smaller = stop_id > time_of_GRB_ISDC_MJD

    try:
        id_number = list(mask_smaller*mask_larger).index(True)
    except:
        raise Exception('No pointing id contains this time...')

    return ids[id_number].decode("utf-8")


def ISDC_MJD(time_object):
    """
    :param time_object: Astropy time object of grb time
    :return: Time in Integral MJD time
    """
    time_object = get_time_object(time_object)
    return time_object.tt.mjd-51544


def ISDC_MJD_to_cxcsec(ISDC_MJD_time):
    """
    Convert ISDC_MJD to UTC
    :param ISDC_MJD_time: time in ISDC_MJD time format
    :return: time in cxcsec format (seconds since 1998-01-01 00:00:00)
    """

    return Time(ISDC_MJD_time+51544, format='mjd', scale='utc').cxcsec
