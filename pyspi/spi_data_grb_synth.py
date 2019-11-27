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
from scipy.integrate import *
#TODO: Data access without afs? 

class SpiData_synthGRB(object):

    def __init__(self, time_of_GRB, response_object, ra=10., dec=10., duration_of_GRB=10, GRB_spectrum_function=None, afs=True, ebounds=None):
        """
        Object if one wants to get the data around a dummy GRB time. A synth GRB is added at 
        this time.
        :param time_of_GRB: Time of GRB in 'YYMMDD HHMMSS' format. UTC!
        :param duration_of_GRB: Duration of synth GRB
        :param GRB_spectrum_function: Function of the maximal GRB spectrum 
        :param afs: Use afs server? 
        :return:
        """
        self._time_of_GRB = time_of_GRB

        if afs:
            print('You chose data access via the afs server')

        # Find path to dir with the needed data (need later spi_oper.fits.gz and sc_orbit_param.fits.gz
        self._pointing_id = self._find_needed_ids(self._time_of_GRB)
        
        if afs:
            
            # Get the data from the afs server
            get_files_afs(self._pointing_id)
            
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
            self._ebounds = None
            self._ene_min = None
            self._ene_max = None

        # Bin data in energy and time
        self.time_and_energy_bin_sgl(time_bin_step=1.)
        
        self.add_GRB_sgl(response_object, t_GRB=duration_of_GRB, F_GRB=GRB_spectrum_function, ra=ra, dec=dec)

    def add_GRB_sgl(self, response_object, F_GRB=None, t_GRB=10, ra=0, dec=0):
         
        assert F_GRB is not None, 'Please give a function for spectrum of GRB'

        #eff_area = response_object.get_binned_effective_area(np.deg2rad(ra),np.deg2rad(dec))
        

        # Flux calculation in bins
        
        Flux_max = np.array([])
        
        for i, (e_l, e_h) in enumerate(zip(self._ebounds[:-1],self._ebounds[1:])):
            
            Flux_max = np.append(Flux_max, trapz([F_GRB(e_l), F_GRB(e_h)],[e_l,e_h]))
            

        #counts_rates_max = np.multiply(Flux_max, eff_area) 

        first_bin = np.argmax(self.time_bins_start[self.time_bins_start<0])
        index_range = first_bin+1 + range(len(self.time_bins_start[np.logical_and(self.time_bins_start>0, self.time_bins_start<t_GRB)]))      
        wgt_time = np.array([])
        np.random.seed(1000)
        for i in index_range:
            if self.time_bins_start[i]<t_GRB/3.:
                wgt_time = np.append(wgt_time, ((self.time_bins_start[i])/(t_GRB/3.))**(4./5.))
            else:
                wgt_time = np.append(wgt_time, ((t_GRB/3.)/(self.time_bins_start[i]))**(4./5.))
        wgt_time = np.ones_like(wgt_time) ##############
        for d in self.energy_and_time_bin_sgl_dict.keys():
            eff_area = response_object.get_binned_effective_area_det(np.deg2rad(ra),np.deg2rad(dec), d)
            for i in index_range:
                self.energy_and_time_bin_sgl_dict[d][i] +=  np.random.poisson(eff_area*Flux_max*wgt_time[i-index_range[0]]*self.time_bin_length[i])
        print('Added GRB from 0 to {} seconds at position ra {} dec {} to binned sgl data'.format(t_GRB, ra, dec))
    
    def time_and_energy_bin_sgl(self, ebounds=None, time_bin_step=1):
        """
        Function that bins the sgl data in energy and time space to use defined bins
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :param time_bin_step: width of the time bins
        """
        self.energy_bin_sgl_data(ebounds)
        self.time_bin_sgl(time_bin_step)
        

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
        self._time_of_GRB_ISDC_MJD = self._string_to_ISDC_MJD(self._time_of_GRB)

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

    def _string_to_ISDC_MJD(self, timestring):
        """
        :param timestring: Time in string format 'YYMMDD HHMMSS'
        :return: Time in Integral MJD time
        """
        
        date = datetime.strptime(timestring, '%y%m%d %H%M%S')
        astro_date = Time(date)
        #TODO Check time. In integral file ISDC_MJD time != DateStart. Which is correct?
        return astro_date.mjd-51544
    def _ISDC_MJD_to_cxcsec(self, ISDC_MJD_time):
        """
        Convert ISDC_MJD to UTC
        :param ISDC_MJD_time: time in ISDC_MJD time format
        :return: time in cxcsec format (seconds since 1998-01-01 00:00:00)
        """
        return Time(ISDC_MJD_time+51544, format='mjd').cxcsec
        
    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id 
        :param pointing_id: pointing_id for which we want the data
        :return:
        """
        
        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:

            # Energy of all events
            energy_sgl = hdu_oper[1].data['energy']
            energy_psd = hdu_oper[2].data['energy']
            energy_me2 = hdu_oper[4].data['energy']
            energy_me3 = hdu_oper[5].data['energy']

            # Time of all events in seconds to or since GRB time
            GRB_ref_time_cxcsec = self._ISDC_MJD_to_cxcsec(self._time_of_GRB_ISDC_MJD)
            
            time_sgl = self._ISDC_MJD_to_cxcsec(hdu_oper[1].data['time'])-GRB_ref_time_cxcsec
            time_psd = self._ISDC_MJD_to_cxcsec(hdu_oper[2].data['time'])-GRB_ref_time_cxcsec
            time_me2 = self._ISDC_MJD_to_cxcsec(hdu_oper[4].data['time'])-GRB_ref_time_cxcsec
            time_me3 = self._ISDC_MJD_to_cxcsec(hdu_oper[5].data['time'])-GRB_ref_time_cxcsec

            self._time_start = np.min([time_sgl[0], time_psd[0], time_me2[0], time_me3[0]])
            self._time_stop = np.max([time_sgl[-1], time_psd[-1], time_me2[-1], time_me3[-1]]) 
            
            # Det of all events
            dets_sgl = hdu_oper[1].data['DETE']
            dets_psd = hdu_oper[2].data['DETE']
            dets_me2 = hdu_oper[4].data['DETE']
            dets_me3 = hdu_oper[5].data['DETE']

        # Build dic with entry for every det

        # For sgl and psd only one det is hit
        
        sgl_energy_dict = {}
        psd_energy_dict = {}
        me2_energy_dict = {}
        me3_energy_dict = {}

        sgl_time_dict = {}
        psd_time_dict = {}
        me2_time_dict = {}
        me3_time_dict = {}
        
        for i in range(19):
            mask = dets_sgl==i
            if True in mask:
                sgl_energy_dict[i] = energy_sgl[dets_sgl==i]
                sgl_time_dict[i] = time_sgl[dets_sgl==i]
            mask = dets_psd==i
            if True in mask:
                psd_energy_dict[i] = energy_psd[dets_psd==i]
                psd_time_dict[i] = time_psd[dets_psd==i]

        # For me2 events two dets are hit
        for i in range(19):
            for k in range(19):
                mask = np.logical_and(dets_me2[:,0]==i, dets_me2[:,1]==k)
                if True in mask:
                    me2_energy_dict['{}-{}'.format(i,k)] = energy_me2[mask]
                    me2_time_dict['{}-{}'.format(i,k)] = time_me2[mask]
                
                #me2_energy_dict['{}-{}'.format(i,k)] = energy_me2[np.logical_or(np.logical_and(dets_me2[:,0]==i, dets_me2[:,1]==k), np.logical_and(dets_me2[:,0]==k, dets_me2[:,1]==i))]
                #me2_time_dict['{}-{}'.format(i,k)] = time_me2[np.logical_or(np.logical_and(dets_me2[:,0]==i, dets_me2[:,1]==k), np.logical_and(dets_me2[:,0]==k, dets_me2[:,1]==i))]

        
        # For me3 events three dets are hit
        for i in range(19):
            for k in range(19):
                for l in range(19):
                    mask = np.logical_and(np.logical_and(dets_me3[:,0]==i, dets_me3[:,1]==k), dets_me3[:,2]==l)
                    if True in mask:
                        me3_energy_dict['{}-{}-{}'.format(i,k,l)] = energy_me3[mask]
                        me3_time_dict['{}-{}-{}'.format(i,k,l)] = time_me3[mask]

        self._sgl_energy_dict = sgl_energy_dict
        self._psd_energy_dict = psd_energy_dict
        self._me2_energy_dict = me2_energy_dict
        self._me3_energy_dict = me3_energy_dict

        self._sgl_time_dict = sgl_time_dict
        self._psd_time_dict = psd_time_dict
        self._me2_time_dict = me2_time_dict
        self._me3_time_dict = me3_time_dict
        
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

        raise NotImplementedError('Not implemented yet. Do not know how to combine the energies')
    
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

        raise NotImplementedError('Not implemented yet. Do not know how to combine the energies')


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

    def time_bin_sgl(self, time_bin_step):
        """
        Bin the already binned in energy data in time bins with constant width
        :param time_bin_step: Width of the time bins
        :return:
        """

        self._time_bins= np.arange(self._time_start, self._time_stop, time_bin_step)[5:-5]
        self._time_bins_start = self._time_bins[:-1]
        self._time_bins_stop = self._time_bins[1:]
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
