import numpy as np
from astropy.time.core import Time
from datetime import datetime
import h5py

from pyspi.io.package_data import get_path_of_data_file
from pyspi.utils.function_utils import get_time_object
from pyspi.utils.detector_ids import double_names, triple_names



def get_live_dets(time, event_types=["single", "double", "triple"]):
    """
    Get the live dets for a given time
    :param time: Live dets at a given time. Either
    "YYMMDD HHMMSS" or as astropy time object
    :param event_types: which event types?
    :return: array of live dets
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
    :return:
    """
    # get end time of pointing
    id_file_path = get_path_of_data_file('id_data_time.hdf5')
    with h5py.File(id_file_path, "r") as f:
        idx = np.argwhere(f["ID"][()] == pointing.encode('utf-8'))
        assert len(idx) != 0, "Poiinting not found in database"
        idx = idx[0,0]

        isdc_mjd_time = f["Start"][idx]

    time = Time(isdc_mjd_time+51544, format='mjd', scale='utc')

    return get_live_dets(time, event_types)
