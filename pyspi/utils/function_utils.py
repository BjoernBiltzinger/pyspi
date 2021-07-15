import numpy as np
from pyspi.io.package_data import get_path_of_data_file
from astropy.time.core import Time, TimeDelta
from datetime import datetime
import h5py

# Collection of needed functions that are used by several classes

def construct_energy_bins(ebounds):
        """
        Function to construct the final energy bins that will be used in the analysis.
        Basically only does one thing: If the single events are included in the analysis
        it ensures that no energybin is covering simultaneously energy outside and inside
        of [psd_low_energy, psd_high_energy]. In this area the single detection photons
        that were not tested by the PSD suffer the "electronical noise" and are very unrealiable.
        The events that have passed the PSD test do not suffer from this "electronical noise".
        Thus we want to only use the PSD events in this energy range. Therefore we construct the
        ebins in such a way that there are ebins outside of this energy range, for which we will
        use normal single + psd events and ebins inside of this energy range for which we only
        want to use the psd events.
        :return:
        """

        psd_low_energy = 1400
        psd_high_energy = 1700

        change = False
        sgl_mask = np.ones(len(ebounds)-1, dtype=bool)
        # Case 1400-1700 is completly in the ebound range
        if ebounds[0] < psd_low_energy \
           and ebounds[-1] > psd_high_energy:
            psd_bin = True
            start_found = False
            stop_found = False
            for i, e in enumerate(ebounds):
                if e >= psd_low_energy and not start_found:
                    start_index = i
                    start_found = True
                if e >= 1700 and not stop_found:
                    stop_index = i
                    stop_found = True
            ebounds = np.insert(ebounds, start_index, psd_low_energy)
            ebounds = np.insert(ebounds, stop_index+1, psd_high_energy)

            if stop_index-start_index > 1:
                sgl_mask = np.logical_and(np.logical_or(ebounds[:-1] <= psd_low_energy,
                                                        ebounds[:-1] >= psd_high_energy),
                                          np.logical_or(ebounds[1:] <= psd_low_energy,
                                                        ebounds[1:] >= psd_high_energy))
            elif stop_index-start_index == 1:
                sgl_mask = np.ones(len(ebounds)-1, dtype=bool)
                sgl_mask[start_index] = False
                sgl_mask[stop_index] = False
            elif stop_index-start_index == 0:
                sgl_mask = np.ones(len(ebounds)-1, dtype=bool)
                sgl_mask[start_index] = False
            change = True
        # Upper bound of erange in psd bin
        elif ebounds[0] < psd_low_energy and ebounds[-1] > psd_low_energy:
            psd_bin = True
            start_found = False
            for i, e in enumerate(ebounds):
                if e >= psd_low_energy and not start_found:
                    start_index = i
                    start_found = True
            ebounds = np.insert(ebounds, start_index, psd_low_energy)
            sgl_mask = (ebounds < psd_low_energy)[:-1]
            change = True
        # Lower bound of erange in psd bin
        elif ebounds[0] < psd_high_energy and ebounds[-1] > psd_high_energy:
            psd_bin = True
            stop_found = False
            for i, e in enumerate(ebounds):
                if e >= psd_high_energy and not stop_found:
                    stop_index = i
                    stop_found = True
            ebounds = np.insert(ebounds, stop_index, psd_high_energy)
            sgl_mask = (ebounds > psd_high_energy)[:-1]
            change = True

        # Or completly in electronic noise range
        elif ebounds[0] > psd_low_energy and ebounds[-1] < psd_high_energy:
            sgl_mask = np.zeros(len(ebounds)-1,dtype=bool)
        # else erange completly outside of psd bin

        return ebounds, sgl_mask

def leapseconds(time_object):
        """
        Hard coded leap seconds from start of INTEGRAL to time of time_object
        :param time_object: Time object to which the number of leapseconds should be detemined
        :return: TimeDelta object of the needed leap seconds
        """
        if time_object < Time(datetime.strptime('060101 000000', '%y%m%d %H%M%S')):
            lsec = 0
        elif time_object < Time(datetime.strptime('090101 000000', '%y%m%d %H%M%S')):
            lsec = 1
        elif time_object < Time(datetime.strptime('120701 000000', '%y%m%d %H%M%S')):
            lsec = 2
        elif time_object < Time(datetime.strptime('150701 000000', '%y%m%d %H%M%S')):
            lsec = 3
        elif time_object < Time(datetime.strptime('170101 000000', '%y%m%d %H%M%S')):
            lsec = 4
        else:
            lsec = 5
        return TimeDelta(lsec, format='sec')

def find_needed_ids(time):
    """
    Get the pointing id of the needed data to cover the GRB time
    :return: Needed pointing id
    """

    # Path to file, which contains id information and start and stop
    # time
    id_file_path = get_path_of_data_file('id_data_time.hdf5')

    # Get GRB time in ISDC_MJD
    time_of_GRB_ISDC_MJD = (time+leapseconds(time)).tt.mjd-51544

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

    #TODO Check time. In integral file ISDC_MJD time != DateStart. Which is correct?
    return (time_object+leapseconds(time_object)).tt.mjd-51544

def ISDC_MJD_to_cxcsec(ISDC_MJD_time):
    """
    Convert ISDC_MJD to UTC
    :param ISDC_MJD_time: time in ISDC_MJD time format
    :return: time in cxcsec format (seconds since 1998-01-01 00:00:00)
    """
    return Time(ISDC_MJD_time+51544, format='mjd', scale='utc').cxcsec
