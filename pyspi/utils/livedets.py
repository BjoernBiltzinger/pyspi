import numpy as np
from astropy.time.core import Time
from datetime import datetime
import h5py
import os

from pyspi.io.package_data import get_path_of_internal_data_dir
from pyspi.utils.function_utils import get_time_object

double_names = {19: [0, 1], 20: [0, 2], 21: [0, 3], 22: [0, 4], 23: [0, 5],
                24: [0, 6], 25: [1, 2], 26: [1, 6], 27: [1, 7], 28: [1, 8],
                29: [1, 9], 30: [2, 3], 31: [2, 9], 32: [2, 10], 33: [2, 11],
                34: [3, 4], 35: [3, 11], 36: [3, 12], 37: [3, 13], 38: [4, 5],
                39: [4, 13], 40: [4, 14], 41: [4, 15], 42: [5, 6], 43: [5, 15],
                44: [5, 16], 45: [5, 17], 46: [6, 7], 47: [6, 17], 48: [6, 18],
                49: [7, 8], 50: [7, 18], 51: [8, 9], 52: [9, 10], 53: [10, 11],
                54: [11, 12], 55: [12, 13], 56: [13, 14], 57: [14, 15],
                58: [15, 16], 59: [16, 17], 60: [17, 18]
                }

triple_names = {61: [0, 1, 2], 62: [0, 2, 3], 63: [0, 3, 4], 64: [0, 4, 5],
                65: [0, 5, 6], 66: [0, 6, 1], 67: [1, 2, 9],
                68: [1, 6, 7], 69: [1, 7, 8], 70: [1, 8, 9], 71: [2, 3, 11],
                72: [2, 9, 10], 73: [2, 10, 11], 74: [3, 4, 13],
                75: [3, 11, 12], 76: [3, 12, 13],
                77: [4, 5, 15], 78: [4, 13, 14],
                79: [4, 14, 15], 80: [5, 6, 17],
                81: [5, 15, 16], 82: [5, 16, 17],
                83: [6, 7, 18], 84: [6, 17, 18]}


def get_live_dets(time, event_types=["single", "double", "triple"]):
    """
    Get the live dets for a given time

    :param time: Live dets at a given time. Either
        "YYMMDD HHMMSS" or as astropy time object
    :param event_types: which event types?
        List with single, double and/or triple

    :returns: array of live dets
    """

    time = get_time_object(time)

    # All single dets
    live_dets = np.arange(19)
    dead_dets = []
    # Check if time is after the failure times
    # (from https://www.isdc.unige.ch/integral/download/osa/doc/10.1/
    # osa_um_spi/node69.html )
    if time > Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets != 2]
        dead_dets.append(2)
    if time > Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets != 17]
        dead_dets.append(17)
    if time > Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets != 5]
        dead_dets.append(5)
    if time > Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets != 1]
        dead_dets.append(1)

    all_dets = np.array([])
    if "single" in event_types:
        all_dets = np.concatenate([all_dets, live_dets])

    if "double" in event_types:
        live_double_dets = []
        for key, value in zip(double_names.keys(),
                              double_names.values()):
            dead = False
            for v in value:
                if v in dead_dets:
                    dead = True

            if not dead:
                live_double_dets.append(key)
        all_dets = np.concatenate([all_dets,
                                   live_double_dets])
    if "triple" in event_types:
        live_triple_dets = []
        for key, value in zip(triple_names.keys(),
                              triple_names.values()):
            dead = False
            for v in value:
                if v in dead_dets:
                    dead = True

            if not dead:
                live_triple_dets.append(key)
        all_dets = np.concatenate([all_dets,
                                   live_triple_dets])
    return np.array(all_dets, dtype=int)


def get_live_dets_pointing(pointing,
                           event_types=["single", "double", "triple"]):
    """
    Get livedets for a given pointing id

    :param pointing: pointing id
    :param event_types: which event types?
        List with single, double and/or triple

    :returns:
    """
    # get end time of pointing
    id_file_path = os.path.join(get_path_of_internal_data_dir,
                                'id_data_time.hdf5')
    with h5py.File(id_file_path, "r") as f:
        idx = np.argwhere(f["ID"][()] == pointing.encode('utf-8'))
        assert len(idx) != 0, "Poiinting not found in database"
        idx = idx[0,0]

        isdc_mjd_time = f["Start"][idx]

    time = Time(isdc_mjd_time+51544, format='mjd', scale='utc')

    return get_live_dets(time, event_types)
