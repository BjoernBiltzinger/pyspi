from astropy.time.core import Time
from datetime import datetime
from pyspi.utils.detector_ids import double_names, triple_names
import numpy as np

def get_live_dets(time, event_types=["single", "double", "triple"]):
    """
    Get the live dets for a given time
    """
    if not isinstance(time, Time):
        time = datetime.strptime(time, '%y%m%d %H%M%S')
        time = Time(time)

    # All single dets
    live_dets = np.arange(19)
    dead_dets = []
    # Check if time is after the failure times
    # (from https://www.isdc.unige.ch/integral/download/osa/doc/10.1/osa_um_spi/node69.html )
    if time>Time(datetime.strptime('031206 060000', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets!=2]
        dead_dets.append(2)
    if time>Time(datetime.strptime('040717 082006', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets!=17]
        dead_dets.append(17)
    if time>Time(datetime.strptime('090219 095957', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets!=5]
        dead_dets.append(5)
    if time>Time(datetime.strptime('100527 124500', '%y%m%d %H%M%S')):
        live_dets = live_dets[live_dets!=1]
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
