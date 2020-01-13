import numpy as np
import astropy.io.fits as fits
import os
from datetime import datetime
from astropy.time.core import Time
from pyspi.io.get_files import get_files_afs, get_files_isdcarc
from pyspi.io.package_data import get_path_of_data_file, get_path_of_external_data_dir
import h5py
from pyspi.utils.progress_bar import progress_bar
import matplotlib.pyplot as plt
from pyspi.utils.detector_ids import double_names, triple_names

class SpiData_GRB(object):

    def __init__(self, time_of_GRB, event_types = ["single"], afs=True, ebounds=None):
        """
        Object if one wants to get the data around a GRB time
        :param time_of_GRB: Time of GRB in 'YYMMDD HHMMSS' format. UTC!
        :param afs: Use afs server? 
        :return:
        """
        self._time_of_GRB = time_of_GRB

        # Which event types are needed?
        self._event_types = event_types
        if afs:
            print('You chose data access via the afs server')

        # Find path to dir with the needed data (need later spi_oper.fits.gz and sc_orbit_param.fits.gz
        self._pointing_id = self._find_needed_ids()
        
        if afs:
            try:
                # Get the data from the afs server
                get_files_afs(self._pointing_id)
            except:
                # Get the files from the iSDC data archive
                print('AFS data access did not work. I will try the ISDC data archive.')
                get_files_isdcarc(self._pointing_id)
        else:
            # Get the files from the iSDC data archive 
            get_files_isdcarc(self._pointing_id)

        # Read in needed data
        self._read_in_pointing_data(self._pointing_id)

        # Save given ebounds
        if ebounds is not None:
            self._ebounds = ebounds
            self._ene_min = ebounds[:-1]
            self._ene_max = ebounds[1:]
        else:
            # Default energy bins
            self._ebounds = np.logspace(np.log10(20),np.log10(8000),30)
            self._ene_min = self._ebounds[:-1]
            self._ene_max = self._ebounds[1:]

    def time_and_energy_bin(self, ebounds=None, time_bin_step=1, start=None, stop=None):
        
        """
        Function that bins the sgl data in energy and time space to use defined bins
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :param time_bin_step: width of the time bins
        """
        if "single" in self._event_types:
            self.time_and_energy_bin_sgl(ebounds, time_bin_step, start, stop)
        if "double" in self._event_types:
            self.time_and_energy_bin_me2(ebounds, time_bin_step, start, stop)
        if "triple" in self._event_types:
            self.time_and_energy_bin_me3(ebounds, time_bin_step, start, stop)
        if "psd" in self._event_types:
            self.time_and_energy_bin_psd(ebounds, time_bin_step, start, stop)
            
    def time_and_energy_bin_sgl(self, ebounds=None, time_bin_step=1, start=None, stop=None):
        """
        Function that bins the sgl data in energy and time space to use defined bins
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :param time_bin_step: width of the time bins
        """
        self.energy_bin_sgl_data(ebounds)

        if start==None or start<self._time_start:
            start = self._time_start
        if stop==None or stop>self._time_stop:
            stop = self._time_stop
            
        self.time_bin_sgl(time_bin_step, start, stop)
        
    def time_and_energy_bin_psd(self, ebounds=None, time_bin_step=1, start=None, stop=None):
        """
        Function that bins the sgl data in energy and time space to use defined bins
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :param time_bin_step: width of the time bins
        """
        self.energy_bin_psd_data(ebounds)

        if start==None or start<self._time_start:
            start = self._time_start
        if stop==None or stop>self._time_stop:
            stop = self._time_stop
            
        self.time_bin_psd(time_bin_step, start, stop)

        
    def time_and_energy_bin_me2(self, ebounds=None, time_bin_step=1, start=None, stop=None):
        """
        Function that bins the sgl data in energy and time space to use defined bins
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :param time_bin_step: width of the time bins
        """
        self.energy_bin_me2_data(ebounds)

        if start==None or start<self._time_start:
            start = self._time_start
        if stop==None or stop>self._time_stop:
            stop = self._time_stop
            
        self.time_bin_me2(time_bin_step, start, stop)

    def time_and_energy_bin_me3(self, ebounds=None, time_bin_step=1, start=None, stop=None):
        """
        Function that bins the sgl data in energy and time space to use defined bins
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :param time_bin_step: width of the time bins
        """
        self.energy_bin_me3_data(ebounds)

        if start==None or start<self._time_start:
            start = self._time_start
        if stop==None or stop>self._time_stop:
            stop = self._time_stop
            
        self.time_bin_me3(time_bin_step, start, stop)


    def set_binned_data_energy_bounds(self, ebounds):
        """
        Change the energy bins for the binned effective_area
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :return:
        """
        
        if not np.array_equal(ebounds, self._ebounds):
            
            print('You have changed the energy boundaries for the binning of the data in the further calculations!')
            self._ene_min = ebounds[:-1]
            self._ene_max = ebounds[1:]
            self._ebounds = ebounds

        
    def _find_needed_ids(self):
        """
        Get the pointing id of the needed data to cover the GRB time 
        :return: Needed pointing id
        """

        # Path to file, which contains id information and start and stop
        # time 
        id_file_path = get_path_of_data_file('id_data_time.hdf5')

        # Get GRB time in ISDC_MJD 
        self._time_of_GRB_ISDC_MJD = self._ISDC_MJD(self._time_of_GRB)

        # Get which id contain the needed time. When the wanted time is
        # too close to the boundarie also add the pervious or following
        # observation id
        id_file = h5py.File(id_file_path,'r')
        start_id = id_file['Start'].value
        stop_id = id_file['Stop'].value
        ids = id_file['ID'].value

        mask_larger = start_id<self._time_of_GRB_ISDC_MJD
        mask_smaller = stop_id>self._time_of_GRB_ISDC_MJD
    
        try:
            id_number = list(mask_smaller*mask_larger).index(True)
            print('Needed data is stored in pointing_id: {}'.format(ids[id_number]))
        except:
            raise Exception('No pointing id contains this time...')
            
        return ids[id_number]

    def _ISDC_MJD(self, time_object):
        """
        :param time_object: Astropy time object of grb time
        :return: Time in Integral MJD time
        """

        #TODO Check time. In integral file ISDC_MJD time != DateStart. Which is correct?
        return time_object.tt.mjd-51544
    
    def _ISDC_MJD_to_cxcsec(self, ISDC_MJD_time):
        """
        Convert ISDC_MJD to UTC
        :param ISDC_MJD_time: time in ISDC_MJD time format
        :return: time in cxcsec format (seconds since 1998-01-01 00:00:00)
        """
        return Time(ISDC_MJD_time+51544, format='mjd', scale='tt').cxcsec
        
    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id 
        :param pointing_id: pointing_id for which we want the data
        :return:
        """
        # Reference time of GRB
        GRB_ref_time_cxcsec = self._ISDC_MJD_to_cxcsec(self._time_of_GRB_ISDC_MJD)
        
        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:
            self._time_start = 10**20
            self._time_stop = -10**20
            # Energy, time and dets of all events
            if "single" in self._event_types:
                energy_sgl = hdu_oper[1].data['energy']
                time_sgl = self._ISDC_MJD_to_cxcsec(hdu_oper[1].data['time'])-GRB_ref_time_cxcsec
                dets_sgl = hdu_oper[1].data['DETE']
                self._time_start=np.min(time_sgl)
                self._time_stop=np.max(time_sgl)
                
            if "psd" in self._event_types:
                energy_psd = hdu_oper[2].data['energy']
                time_psd = self._ISDC_MJD_to_cxcsec(hdu_oper[2].data['time'])-GRB_ref_time_cxcsec
                dets_psd = hdu_oper[2].data['DETE']
                if np.min(time_psd)<self._time_start:
                    self._time_start=np.min(time_psd)
                if np.max(time_psd)>self._time_stop:
                    self._time_stop=np.max(time_psd)
                    
            if "double" in self._event_types:
                energy_me2 = hdu_oper[4].data['energy']
                time_me2 = self._ISDC_MJD_to_cxcsec(hdu_oper[4].data['time'])-GRB_ref_time_cxcsec
                dets_me2 = hdu_oper[4].data['DETE']
                if np.min(time_me2)<self._time_start:
                    self._time_start=np.min(time_me2)
                if np.max(time_me2)>self._time_stop:
                    self._time_stop=np.max(time_me2)
                
            if "triple" in self._event_types:
                energy_me3 = hdu_oper[5].data['energy']
                time_me3 = self._ISDC_MJD_to_cxcsec(hdu_oper[5].data['time'])-GRB_ref_time_cxcsec
                dets_me3 = hdu_oper[5].data['DETE']
                if np.min(time_me3)<self._time_start:
                    self._time_start=np.min(time_me3)
                if np.max(time_me3)>self._time_stop:
                    self._time_stop=np.max(time_me3)
            
        # Build dic with entry for every det (0-84)
        # For sgl and psd only one det is hit

        if "single" in self._event_types:
            sgl_energy_dict = {}
            sgl_time_dict = {}
            for i in range(19):
                mask = dets_sgl==i
                #if True in mask:
                sgl_energy_dict[i] = energy_sgl[mask]
                sgl_time_dict[i] = time_sgl[mask]
                
            self._sgl_energy_dict = sgl_energy_dict
            self._sgl_time_dict = sgl_time_dict

            # Get a list with the dets that seem to be defect (because there are 0 counts in them)
            self._bad_sgl_dets = np.zeros(19, dtype=bool)
            for i in range(19):
                if sgl_energy_dict[i].size==0:
                    self._bad_sgl_dets[i] = True

        # PSD events
        if "psd" in self._event_types:
            psd_energy_dict = {}
            psd_time_dict = {}
            for i in range(19):
                mask = dets_psd==i
                psd_energy_dict[i] = energy_psd[mask]
                psd_time_dict[i] = time_psd[mask]
                
            self._psd_energy_dict = psd_energy_dict
            self._psd_time_dict = psd_time_dict

            # Get a list with the dets that seem to be defect (because there are 0 counts in them)
            self._bad_psd_dets = np.zeros(19, dtype=bool)
            for i in range(19):
                if psd_energy_dict[i].size==0:
                    self._bad_psd_dets[i] = True
            
        # Double events
        if "double" in self._event_types:
            me2_energy_dict = {}
            me2_time_dict = {}
            # For me2 events two dets are hit
            for n, (i, k) in enumerate(double_names.values(), start=19):
                    mask1 = np.logical_and(dets_me2[:,0]==i, dets_me2[:,1]==k)
                    mask2 = np.logical_and(dets_me2[:,0]==k, dets_me2[:,1]==i)

                    e_array1 = energy_me2[mask1]
                    e_array2 = energy_me2[mask2]

                    t_array1 = time_me2[mask1]
                    t_array2 = time_me2[mask2]

                    total_e_array = np.concatenate((e_array1, e_array2))
                    total_t_array = np.concatenate((t_array1, t_array2))

                    # time sort mask
                    mask = np.argsort(total_t_array)

                    me2_energy_dict[n] = np.sum(total_e_array[mask], axis=1)
                    me2_time_dict[n] = total_t_array[mask]

            self._me2_energy_dict = me2_energy_dict
            self._me2_time_dict = me2_time_dict

            # Get a list with the dets that seem to be defect (because there are 0 counts in them)
            self._bad_me2_dets = np.zeros(42, dtype=bool)
            for i in range(19,61):
                if me2_energy_dict[i].size==0:
                    self._bad_me2_dets[i-19] = True

        # Triple events
        if "triple" in self._event_types:
            me3_energy_dict = {}
            me3_time_dict = {}
            # For me3 events three dets are hit
            for n, (i, j, k) in enumerate(triple_names.values(), start=61):
                mask1 = np.logical_and(np.logical_and(dets_me3[:,0]==i, dets_me3[:,1]==j), dets_me3[:,2]==k)
                mask2 = np.logical_and(np.logical_and(dets_me3[:,0]==i, dets_me3[:,1]==k), dets_me3[:,2]==j)
                mask3 = np.logical_and(np.logical_and(dets_me3[:,0]==j, dets_me3[:,1]==i), dets_me3[:,2]==k)
                mask4 = np.logical_and(np.logical_and(dets_me3[:,0]==j, dets_me3[:,1]==k), dets_me3[:,2]==i)
                mask5 = np.logical_and(np.logical_and(dets_me3[:,0]==k, dets_me3[:,1]==i), dets_me3[:,2]==j)
                mask6 = np.logical_and(np.logical_and(dets_me3[:,0]==k, dets_me3[:,1]==j), dets_me3[:,2]==i)

                e_array1 = energy_me3[mask1]
                e_array2 = energy_me3[mask2]
                e_array3 = energy_me3[mask3]
                e_array4 = energy_me3[mask4]
                e_array5 = energy_me3[mask5]
                e_array6 = energy_me3[mask6]

                t_array1 = time_me3[mask1]
                t_array2 = time_me3[mask2]
                t_array3 = time_me3[mask3]
                t_array4 = time_me3[mask4]
                t_array5 = time_me3[mask5]
                t_array6 = time_me3[mask6]

                total_e_array = np.concatenate((e_array1, e_array2, e_array3, e_array4, e_array5, e_array6))
                total_t_array = np.concatenate((t_array1, t_array2, t_array3, t_array4, t_array5, t_array6))

                # time sort mask
                mask = np.argsort(total_t_array) 

                me3_energy_dict[n] = np.sum(total_e_array[mask], axis=1)
                me3_time_dict[n] = total_t_array[mask]

            self._me3_energy_dict = me3_energy_dict
            self._me3_time_dict = me3_time_dict

            # Get a list with the dets that seem to be defect (because there are 0 counts in them)
            self._bad_me3_dets = np.zeros(24, dtype=bool)
            for i in range(61,85):
                if me3_energy_dict[i].size==0:
                    self._bad_me3_dets[i-61] = True

    def energy_bin_all_data(self, ebounds=None):
        """
        Function to bin all data (sgl, psd, me2 and me3) in user defined energy bins
        :param ebounds: Specify new ebounds for the bins. If None than default values saved in the 
        object are used.
        :return:
        """
        self.energy_bin_sgl_data(ebounds)
        self.energy_bin_psd_data(ebounds)
        #self.energy_bin_me2_data(ebounds)
        #self.energy_bin_me3_data(ebounds)

            
    def energy_bin_sgl_data(self, ebounds=None):
        """
        Function to bin the sgl data in user defined energy bins
        :param ebounds: Specify new ebounds for the bins. If None than default values saved in the 
        object are used.
        :return:
        """

        # Update ebounds if new one is given
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)


        energy_bin_sgl_dict = {}

        # Loop over ebins - TODO: Speed up with mpi4py!
        with progress_bar(len(self.energy_sgl_dict), title='Calculating the bin arrays of the events') as p:
            for d in self.energy_sgl_dict.keys():
                energy_bin_sgl = -np.ones_like(self.energy_sgl_dict[d])
        
                for i in range(len(self._ene_min)):
                    # Single events
                    energy_bin_sgl = np.where(np.logical_and(self.energy_sgl_dict[d]>=self._ene_min[i], self.energy_sgl_dict[d]<self._ene_max[i]), i, energy_bin_sgl)
                energy_bin_sgl_dict[d] = energy_bin_sgl
                p.increase()

        self._energy_bin_sgl_dict = energy_bin_sgl_dict

    def energy_bin_psd_data(self, ebounds=None):
        """
        Function to bin the psd data in user defined energy bins
        :param ebounds: Specify new ebounds for the bins. If None than default values saved in the 
        object are used.
        :return:
        """

        # Update ebounds if new one is given
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)

        energy_bin_psd_dict = {}

        # Loop over ebins - TODO: Speed up with mpi4py!
        with progress_bar(len(self.energy_psd_dict), title='Calculating the bin arrays of the events') as p:
            for d in self.energy_psd_dict.keys():
                energy_bin_psd = -np.ones_like(self.energy_psd_dict[d])
        
                for i in range(len(self._ene_min)):
                    # Single events
                    energy_bin_psd = np.where(np.logical_and(self.energy_psd_dict[d]>=self._ene_min[i], self.energy_psd_dict[d]<self._ene_max[i]), i, energy_bin_psd)
                energy_bin_psd_dict[d] = energy_bin_psd
                p.increase()

        self._energy_bin_psd_dict = energy_bin_psd_dict

    def energy_bin_me2_data(self, ebounds=None):
        """
        Function to bin the me2 data in user defined energy bins
        :param ebounds: Specify new ebounds for the bins. If None than default values saved in the 
        object are used.
        :return:
        """

        # Update ebounds if new one is given
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)
            
        energy_bin_me2_dict = {}

        # Loop over ebins - TODO: Speed up with mpi4py!
        with progress_bar(len(self.energy_me2_dict), title='Calculating the bin arrays of the events') as p:
            for d in self.energy_me2_dict.keys():
                energy_bin_me2 = -np.ones_like(self.energy_me2_dict[d])
        
                for i in range(len(self._ene_min)):
                    # Single events
                    energy_bin_me2 = np.where(np.logical_and(self.energy_me2_dict[d]>=self._ene_min[i], self.energy_me2_dict[d]<self._ene_max[i]), i, energy_bin_me2)
                energy_bin_me2_dict[d] = energy_bin_me2
                p.increase()

        self._energy_bin_me2_dict = energy_bin_me2_dict


    def energy_bin_me3_data(self, ebounds=None):
        """
        Function to bin the me3 data in user defined energy bins
        :param ebounds: Specify new ebounds for the bins. If None than default values saved in the 
        object are used.
        :return:
        """

        # Update ebounds if new one is given
        if ebounds is not None:
            self.set_binned_data_energy_bounds(ebounds)

        energy_bin_me3_dict = {}

        # Loop over ebins - TODO: Speed up with mpi4py!
        with progress_bar(len(self.energy_me3_dict), title='Calculating the bin arrays of the events') as p:
            for d in self.energy_me3_dict.keys():
                energy_bin_me3 = -np.ones_like(self.energy_me3_dict[d])
        
                for i in range(len(self._ene_min)):
                    # Single events
                    energy_bin_me3 = np.where(np.logical_and(self.energy_me3_dict[d]>=self._ene_min[i], self.energy_me3_dict[d]<self._ene_max[i]), i, energy_bin_me3)
                energy_bin_me3_dict[d] = energy_bin_me3
                p.increase()

        self._energy_bin_me3_dict = energy_bin_me3_dict

    def time_bin_sgl(self, time_bin_step, start, stop):
        """
        Bin the already binned in energy data in time bins with constant width
        :param time_bin_step: Width of the time bins
        :return:
        """
        
        self._time_bins = np.array([np.arange(start, stop, time_bin_step)[:-1],
                                    np.arange(start, stop, time_bin_step)[1:]]).T
        self._time_bins_start = self._time_bins[:,0]
        self._time_bins_stop = self._time_bins[:, 1]
        self._time_bin_length = self._time_bins_stop-self._time_bins_start

        energy_and_time_bin_sgl_dict = {}
        
        for d in self.energy_bin_sgl_dict.keys():
            counts_time_energy_binned = np.zeros((len(self._time_bins_start), len(self.ene_min)))
            
            for nb in range(len(self.ene_min)):
                times_energy_bin_events = self.time_sgl_dict[d][self.energy_bin_sgl_dict[d]==nb]
                
                for i in range(len(self._time_bins_start)):
                    counts_time_energy_binned[i,nb] = len(times_energy_bin_events[np.logical_and(times_energy_bin_events>=self._time_bins_start[i], times_energy_bin_events<self._time_bins_stop[i])])
                    
            energy_and_time_bin_sgl_dict[d] = counts_time_energy_binned

        self._energy_and_time_bin_sgl_dict = energy_and_time_bin_sgl_dict

    def time_bin_psd(self, time_bin_step, start, stop):
        """
        Bin the already binned in energy data in time bins with constant width
        :param time_bin_step: Width of the time bins
        :return:
        """
        
        self._time_bins = np.array([np.arange(start, stop, time_bin_step)[:-1],
                                    np.arange(start, stop, time_bin_step)[1:]]).T
        self._time_bins_start = self._time_bins[:,0]
        self._time_bins_stop = self._time_bins[:, 1]
        self._time_bin_length = self._time_bins_stop-self._time_bins_start

        energy_and_time_bin_psd_dict = {}
        
        for d in self.energy_bin_psd_dict.keys():
            counts_time_energy_binned = np.zeros((len(self._time_bins_start), len(self.ene_min)))
            
            for nb in range(len(self.ene_min)):
                times_energy_bin_events = self.time_psd_dict[d][self.energy_bin_psd_dict[d]==nb]
                
                for i in range(len(self._time_bins_start)):
                    counts_time_energy_binned[i,nb] = len(times_energy_bin_events[np.logical_and(times_energy_bin_events>=self._time_bins_start[i], times_energy_bin_events<self._time_bins_stop[i])])
                    
            energy_and_time_bin_psd_dict[d] = counts_time_energy_binned

        self._energy_and_time_bin_psd_dict = energy_and_time_bin_psd_dict

    def time_bin_me2(self, time_bin_step, start, stop):
        """
        Bin the already binned in energy data in time bins with constant width
        :param time_bin_step: Width of the time bins
        :return:
        """
        
        self._time_bins = np.array([np.arange(start, stop, time_bin_step)[:-1],
                                    np.arange(start, stop, time_bin_step)[1:]]).T
        self._time_bins_start = self._time_bins[:,0]
        self._time_bins_stop = self._time_bins[:, 1]
        self._time_bin_length = self._time_bins_stop-self._time_bins_start

        energy_and_time_bin_me2_dict = {}
        
        for d in self.energy_bin_me2_dict.keys():
            counts_time_energy_binned = np.zeros((len(self._time_bins_start), len(self.ene_min)))
            
            for nb in range(len(self.ene_min)):
                times_energy_bin_events = self.time_me2_dict[d][self.energy_bin_me2_dict[d]==nb]
                
                for i in range(len(self._time_bins_start)):
                    counts_time_energy_binned[i,nb] = len(times_energy_bin_events[np.logical_and(times_energy_bin_events>=self._time_bins_start[i], times_energy_bin_events<self._time_bins_stop[i])])
                    
            energy_and_time_bin_me2_dict[d] = counts_time_energy_binned

        self._energy_and_time_bin_me2_dict = energy_and_time_bin_me2_dict

    def time_bin_me3(self, time_bin_step, start, stop):
        """
        Bin the already binned in energy data in time bins with constant width
        :param time_bin_step: Width of the time bins
        :return:
        """
        
        self._time_bins = np.array([np.arange(start, stop, time_bin_step)[:-1],
                                    np.arange(start, stop, time_bin_step)[1:]]).T
        self._time_bins_start = self._time_bins[:,0]
        self._time_bins_stop = self._time_bins[:, 1]
        self._time_bin_length = self._time_bins_stop-self._time_bins_start

        energy_and_time_bin_me3_dict = {}
        
        for d in self.energy_bin_me3_dict.keys():
            counts_time_energy_binned = np.zeros((len(self._time_bins_start), len(self.ene_min)))
            
            for nb in range(len(self.ene_min)):
                times_energy_bin_events = self.time_me3_dict[d][self.energy_bin_me3_dict[d]==nb]
                
                for i in range(len(self._time_bins_start)):
                    counts_time_energy_binned[i,nb] = len(times_energy_bin_events[np.logical_and(times_energy_bin_events>=self._time_bins_start[i], times_energy_bin_events<self._time_bins_stop[i])])
                    
            energy_and_time_bin_me3_dict[d] = counts_time_energy_binned

        self._energy_and_time_bin_me3_dict = energy_and_time_bin_me3_dict

    def plot_binned_sgl_data_ebin(self, ebin=0, det=0, savepath=None):
        """
        Function to plot the binned data of one of the echans and one det.
        :param ebin: Which ebin?
        :param det: Which det?
        :param savepath: Where to save the figure
        :return: figure
        """

        assert det in self.energy_and_time_bin_sgl_dict.keys(), 'The det you want to use is either unvalid or broken! Please use one of these: {}.'.format(self.energy_and_time_bin_sgl_dict.keys())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter((self._time_bins_stop+self.time_bins_start)/2, self.energy_and_time_bin_sgl_dict[det][:, ebin], s=1, facecolors='none', edgecolors='black', alpha=0.4)

        ax.set_ylabel('Count rate [1/s]')
        ax.set_xlabel('Time Since GRB [s]')
        ax.set_title('Det {} | Energy {}-{} keV'.format(det, round(self._ene_min[ebin], 2), round(self._ene_max[ebin],2)))
        fig.tight_layout()
        
        if savepath is not None:
            fig.savefig(savepath)

        return fig

    def plot_binned_sgl_data(self, det=0, savepath=None):
        """
        Function to plot the binned data of all echans and one det.
        :param det: Which det?
        :param savepath: Where to save the figure
        :return: figure
        """

        assert det in self.energy_and_time_bin_sgl_dict.keys(), 'The det you want to use is either unvalid or broken! Please use one of these: {}.'.format(self.energy_and_time_bin_sgl_dict.keys())

        fig = plt.figure()
        ax = fig.add_subplot(111)
        
        ax.scatter((self._time_bins_stop+self.time_bins_start)/2, np.sum(self.energy_and_time_bin_sgl_dict[det], axis=1), s=1, facecolors='none', edgecolors='black', alpha=0.4)

        ax.set_ylabel('Count rate [1/s]')
        ax.set_xlabel('Time Since GRB [s]')
        ax.set_title('Det {} | All Energies'.format(det))
        fig.tight_layout()
        
        if savepath is not None:
            fig.savefig(savepath)

        return fig

    
    @property
    def energy_sgl_dict(self):
        """
        Dict with the energies of all sgl events sorted in the different detectors
        """
        return self._sgl_energy_dict

    @property
    def energy_psd_dict(self):
        return self._psd_energy_dict

    @property
    def energy_me2_dict(self):
        return self._me2_energy_dict

    @property
    def energy_me3_dict(self):
        return self._me3_energy_dict
    
    @property
    def time_sgl_dict(self):
        """
        Dict with the time of the events in energy_sgl_dict
        """
        return self._sgl_time_dict

    @property
    def time_psd_dict(self):
        return self._psd_time_dict

    @property
    def time_me2_dict(self):
        return self._me2_time_dict

    @property
    def time_me3_dict(self):
        return self._me3_time_dict

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
    def energy_bin_sgl_dict(self):
        """
        Dict with the number of the energy bin of the events in energy_sgl_dict
        """
        return self._energy_bin_sgl_dict

    @property
    def energy_bin_psd_dict(self):
        return self._energy_bin_psd_dict

    @property
    def energy_bin_me2_dict(self):
        return self._energy_bin_me2_dict

    @property
    def energy_bin_me3_dict(self):
        return self._energy_bin_me3_dict

    @property
    def energy_and_time_bin_sgl_dict(self):
        return self._energy_and_time_bin_sgl_dict

    @property
    def energy_and_time_bin_psd_dict(self):
        return self._energy_and_time_bin_psd_dict

    @property
    def energy_and_time_bin_me2_dict(self):
        return self._energy_and_time_bin_me2_dict

    @property
    def energy_and_time_bin_me3_dict(self):
        return self._energy_and_time_bin_me3_dict

    @property
    def ebounds(self):
        return self._ebounds

    @property
    def ene_min(self):
        return self._ene_min

    @property
    def ene_max(self):
        return self._ene_max    

    @property
    def geometry_file_path(self):
        return os.path.join(get_path_of_external_data_dir(), 'pointing_data', self._pointing_id, 'sc_orbit_param.fits.gz')
        
    @property
    def grb_time_ISDC_MJD(self):
        return self._time_of_GRB_ISDC_MJD

    @property
    def time_bins(self):
        return self._time_bins

    @property
    def time_bins_start(self):
        return self._time_bins_start

    @property
    def time_bins_stop(self):
        return self._time_bins_stop

    @property
    def time_bin_length(self):
        return self._time_bin_length

    @property
    def sgl_dets_working(self):
        return self.energy_sgl_dict().keys()

    @property
    def bad_sgl_dets(self):
        return self._bad_sgl_dets

    @property
    def bad_psd_dets(self):
        return self._bad_psd_dets

    @property
    def bad_me2_dets(self):
        return self._bad_me2_dets

    @property
    def bad_me3_dets(self):
        return self._bad_me3_dets

    
