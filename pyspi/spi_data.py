import numpy as np
import astropy.io.fits as fits
import os
from datetime import datetime
from astropy.time.core import Time
from pyspi.io.get_files import get_files_afs
from pyspi.io.package_data import get_path_of_data_file, get_path_of_external_data_dir
import h5py

#TODO: Data access without afs? 

class SpiData_GRB(object):

    def __init__(self, time_of_GRB, afs=True):
        """
        Object if one wants to get the data around a GRB time
        :param time_of_GRB: Time of GRB in 'YYMMDD HHMMSS' format. UTC!
        :param afs: Use afs server? 
        :return:
        """
        self._time_of_GRB = time_of_GRB

        if not afs:
            raise NotImplementedError('Only AFS access possible at the moment')
        
        if afs:
            # Find path to dir with the needed data (need later spi_oper.fits.gz and sc_orbit_param.fits.gz
            self._pointing_id = self._find_needed_ids(self._time_of_GRB)

            # Get the data from the afs server
            get_files_afs(self._pointing_id)
            
            # Read in needed data
            self._read_in_pointing_data(self._pointing_id)

    def _find_needed_ids(self, time):
        """
        Get the pointing id of the needed data to cover time 
        :param time: time of interest
        :return: Needed pointing id
        """

        # Path to file, which contains id information and start and stop
        # time 
        id_file_path = get_path_of_data_file('id_data_time.hdf5')

        # Get GRB time in ISDC_MJD 
        time_of_GRB_ISDC_MJD = self._string_to_ISDC_MJD(self._time_of_GRB)

        # Get which id contain the needed time. When the wanted time is
        # too close to the boundarie also add the pervious or following
        # observation id
        id_file = h5py.File(id_file_path,'r')
        start_id = id_file['Start'].value
        stop_id = id_file['Stop'].value
        ids = id_file['ID'].value
        
        mask_larger = start_id<time_of_GRB_ISDC_MJD
        mask_smaller = stop_id>time_of_GRB_ISDC_MJD
    
        try:
            id_number = list(mask_smaller*mask_larger).index(True)
            print('Needed data is stored in pointing_id: {}'.format(ids[id_number]))
        except:
            raise Exception('No pointing id contains this time...')
            
        return ids[id_number]

    def _string_to_ISDC_MJD(self, timestring):
        """
        :param timestring: Time in string format 'YYMMDD HHMMSS'
        :return: Time in Integral MJD time
        """
        
        date = datetime.strptime(timestring, '%y%m%d %H%M%S')
        astro_date = Time(date)
        #TODO Check time. In integral file ISDC_MJD time != DateStart. Which is correct?
        return astro_date.mjd-51544
        
    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id 
        :param pointing_id: pointing_id for which we want the data
        :return:
        """
        
        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:

            # Energy of events
            self._energy_sgl = hdu_oper[1].data['energy']
            self._energy_psd = hdu_oper[2].data['energy']
            self._energy_me2 = hdu_oper[4].data['energy']
            self._energy_me3 = hdu_oper[5].data['energy']

            # Time of events
            self._time_sgl = hdu_oper[1].data['time']
            self._time_psd = hdu_oper[2].data['time']
            self._time_me2 = hdu_oper[4].data['time']
            self._time_me3 = hdu_oper[5].data['time']

            # Det of events
            self._dets_sgl = hdu_oper[1].data['DETE']
            self._dets_psd = hdu_oper[2].data['DETE']
            self._dets_me2 = hdu_oper[4].data['DETE']
            self._dets_me3 = hdu_oper[5].data['DETE']

    @property
    def energy_sgl(self):
        return self._energy_sgl

    @property
    def energy_psd(self):
        return self._energy_psd

    @property
    def energy_me2(self):
        return self._energy_me2

    @property
    def energy_me3(self):
        return self._energy_me3

    @property
    def time_sgl(self):
        return self._time_sgl

    @property
    def time_psd(self):
        return self._time_psd

    @property
    def time_me2(self):
        return self._time_me2

    @property
    def time_me3(self):
        return self._time_me3

    @property
    def dets_sgl(self):
        return self._dets_sgl

    @property
    def dets_psd(self):
        return self._dets_psd

    @property
    def dets_me2(self):
        return self._dets_me2

    @property
    def dets_me3(self):
        return self._dets_me3

    @property
    def geometry_file_path(self):
        return os.path.join(get_path_of_external_data_dir(), 'pointing_data', self._pointing_id, 'sc_orbit_param.fits.gz')
        
