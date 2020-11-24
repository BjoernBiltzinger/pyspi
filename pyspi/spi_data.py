import numpy as np
import astropy.io.fits as fits
import os
from datetime import datetime
from astropy.time.core import Time, TimeDelta
from pyspi.io.get_files import get_files_afs, get_files_isdcarc
from pyspi.io.package_data import get_path_of_data_file, get_path_of_external_data_dir
import h5py
from pyspi.utils.progress_bar import progress_bar
import matplotlib.pyplot as plt
from pyspi.utils.detector_ids import double_names, triple_names
from scipy.integrate import trapz
from threeML.utils.time_series.event_list import EventList

class DataGRB(object):

    def __init__(self, time_of_GRB, detector, afs=True, ebounds=None, use_psd=True):
        """
        Object if one wants to get the data around a GRB time
        :param time_of_GRB: Time of GRB in 'YYMMDD HHMMSS' format. UTC!
        :param afs: Use afs server? 
        :return:
        """
        self._time_of_GRB = time_of_GRB

        self._det = detector
        self._use_psd = use_psd
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

        # TODO This is not very clever. We cover a way to big time than we need.
        # But I am not sure yet how much will be used in the bkg polynominal approach.
        # Maybe change this later to grb_time +- 200 seconds
        self.energy_and_time_bin_data(1, self._time_start, self._time_stop)

    def set_binned_data_energy_bounds(self, ebounds):
        """
        Change the energy bins for the binned effective_area
        :param ebounds: New ebinedges: ebounds[:-1] start of ebins, ebounds[1:] end of ebins
        :return:
        """
        
        if not np.array_equal(ebounds, self._ebounds):
            
            print('You have changed the energy boundaries for the binning of'\
                  ' the data in the further calculations!')
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
        id_file = h5py.File(id_file_path, 'r')
        start_id = id_file['Start'].value
        stop_id = id_file['Stop'].value
        ids = id_file['ID'].value

        mask_larger = start_id < self._time_of_GRB_ISDC_MJD
        mask_smaller = stop_id > self._time_of_GRB_ISDC_MJD
    
        try:
            id_number = list(mask_smaller*mask_larger).index(True)
            print('Needed data is stored in pointing_id: {}'.format(ids[id_number]))
        except:
            raise Exception('No pointing id contains this time...')
            
        return ids[id_number].decode("utf-8")

    def _ISDC_MJD(self, time_object):
        """
        :param time_object: Astropy time object of grb time
        :return: Time in Integral MJD time
        """

        #TODO Check time. In integral file ISDC_MJD time != DateStart. Which is correct?
        return (time_object+self._leapseconds(time_object)).tt.mjd-51544

    def _leapseconds(self, time_object):
        """
        Hard coded leap seconds from start of INTEGRAL to time of time_object
        :param time_object: Time object to which the number of leapseconds should be detemined
        :return: TimeDelta object of the needed leap seconds
        """
        if time_object<Time(datetime.strptime('060101 000000', '%y%m%d %H%M%S')):
            lsec = 0
        elif time_object<Time(datetime.strptime('090101 000000', '%y%m%d %H%M%S')):
            lsec = 1
        elif time_object<Time(datetime.strptime('120701 000000', '%y%m%d %H%M%S')):
            lsec = 2
        elif time_object<Time(datetime.strptime('150701 000000', '%y%m%d %H%M%S')):
            lsec = 3
        elif time_object<Time(datetime.strptime('170101 000000', '%y%m%d %H%M%S')):
            lsec = 4
        else:
            lsec=5
        return TimeDelta(lsec, format='sec')
    
    def _ISDC_MJD_to_cxcsec(self, ISDC_MJD_time):
        """
        Convert ISDC_MJD to UTC
        :param ISDC_MJD_time: time in ISDC_MJD time format
        :return: time in cxcsec format (seconds since 1998-01-01 00:00:00)
        """
        return Time(ISDC_MJD_time+51544, format='mjd', scale='utc').cxcsec
        
    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id 
        :param pointing_id: pointing_id for which we want the data
        :return:
        """
        # Reference time of GRB
        GRB_ref_time_cxcsec = self._ISDC_MJD_to_cxcsec(self._time_of_GRB_ISDC_MJD)
        
        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:

            # Get time of first and last event (t0 at grb time)
            time_sgl = self._ISDC_MJD_to_cxcsec(hdu_oper[1].data['time']) - GRB_ref_time_cxcsec
            time_psd = self._ISDC_MJD_to_cxcsec(hdu_oper[2].data['time']) - GRB_ref_time_cxcsec
            time_me2 = self._ISDC_MJD_to_cxcsec(hdu_oper[4].data['time']) - GRB_ref_time_cxcsec
            time_me3 = self._ISDC_MJD_to_cxcsec(hdu_oper[5].data['time']) - GRB_ref_time_cxcsec

            self._time_start = np.min(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))
            self._time_stop = np.max(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))

            if self._det in range(19):
                dets_sgl = hdu_oper[1].data['DETE']
                time_sgl = time_sgl[dets_sgl == self._det]
                energy_sgl = hdu_oper[1].data['energy'][dets_sgl == self._det]

                #if "psd" in self._event_types:
                if self._use_psd:
                    dets_psd = hdu_oper[2].data['DETE']
                    time_psd = time_psd[dets_psd == self._det]
                    energy_psd = hdu_oper[2].data['energy'][dets_psd == self._det]

            if self._det in range(19, 61):
                dets_me2 = np.sort(hdu_oper[4].data['DETE'], axis=1)
                i, k = double_names[self._det-19]
                mask = np.logical_and(dets_me2[:, 0]==i,
                                      dets_me2[:, 1] == k)

                time_me2 = time_me2[mask]
                energy_me2 = np.sum(hdu_oper[4].data['energy'][mask], axis=1)
                
            if self._det in range(61,85):
                dets_me3 = np.sort(hdu_oper[5].data['DETE'], axis=1)
                i, j, k = tripple_names[self._det-61]
                mask = np.logical_and(np.logical_and(dets_me3[:, 0] == i,
                                                     dets_me3[:, 1] == j),
                                      dets_me3[:, 2] == k)

                time_me3 = time_me3[mask]
                energy_me3 = np.sum(hdu_oper[5].data['energy'][mask], axis=1)

        if self._det in range(19):

            self._times = time_sgl
            self._energies = energy_sgl

            if self._use_psd:

                self._times_psd = time_psd
                self._energies_psd = energy_psd

        if self._det in range(19, 61):

            self._times = time_me2
            self._energies = energy_me2

        if self._det in range(61,85):

            self._times = time_me3
            self._energies = energy_me3

        if np.sum(self._energies)==0:

            raise AssertionError(f"The detector {self._det} has zero counts and is therefore not active."\
                                 "Please exclude this detector!")


    def energy_and_time_bin_data(self, time_bin_step, start, stop):
        """
        Bin the already binned in energy data in time bins with constant width
        :param time_bin_step: Width of the time bins
        :return:
        """
        
        self._time_bins = np.array([np.arange(start, stop, time_bin_step)[:-1],
                                    np.arange(start, stop, time_bin_step)[1:]]).T
        self._time_bins_start = self._time_bins[:,0]
        self._time_bins_stop = self._time_bins[:, 1]
        self._time_bin_edges = np.append(self._time_bins_start,
                                         self._time_bins_stop[-1])
        self._time_bin_length = self._time_bins_stop-self._time_bins_start

        #counts_time_energy_binned = np.zeros((len(self._time_bins_start), len(self.ene_min)))
        self._counts_time_energy_binned, _, _ = np.histogram2d(self._times,
                                                               self._energies,
                                                               bins=[self._time_bin_edges,
                                                                     self._ebounds])

        if self._det in range(19) and self._use_psd:
            self._counts_time_energy_binned_psd, _, _ = np.histogram2d(self._times_psd,
                                                                       self._energies_psd,
                                                                       bins=[self._time_bin_edges,
                                                                             self._ebounds])

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

    def plot_binned_sgl_data_all_dets_together(self, savepath=None):
        """
        Function to plot the binned data of all echans and one det.
        :param det: Which det?
        :param savepath: Where to save the figure
        :return: figure
        """

        fig = plt.figure()
        ax = fig.add_subplot(111)

        total = np.zeros_like(self.energy_and_time_bin_sgl_dict[0])
        for i in range(19):
            total += self.energy_and_time_bin_sgl_dict[i]

        ax.step((self._time_bins_stop+self.time_bins_start)/2, np.sum(total, axis=1), color="black", alpha=1)

        ax.set_ylabel('Count rate [1/s]')
        ax.set_xlabel('Time Since GRB [s]')
        ax.set_title('All dets | All Energies')
        fig.tight_layout()

        if savepath is not None:
            fig.savefig(savepath)

        return fig

    @property
    def counts_time_energy_binned(self):
        return self._counts_time_energy_binned

    @property
    def counts_time_energy_binned_psd(self):
        return self._counts_time_energy_binned_psd

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
    
class DataGRBSimulate(DataGRB):

    def __init__(self, time_of_GRB, response_object, ra=10., dec=10.,
                 duration_of_GRB=10, shapes=None, psd_eff=0.85, afs=True,
                 ebounds=None, event_types=["single"]):
        """
        Data object with an simulated GRB added to a real SPI data file.
        :param time_of_GRB: Time of simulated GRB in 'YYMMDD HHMMSS' format. UTC!
        :param response_object: Response object for this pointing. Needed to simulate GRB.
        :param ra: Ra coordinates of simulated GRB in sat coord. (deg)
        :param dec: Dec coordinates of simulated GRB in sat coord. (deg)
        :param duration_of_GRB: Duration of synth GRB (s)
        :param shapes: Functions for the different source components
        :param psd_edd: PSD effectivity
        :param afs: Use afs server?
        :return:
        """

        super(DataGRBSimulate, self).__init__(time_of_GRB, event_types, afs, ebounds)

        self._response_object = response_object
        self._shapes = shapes
        self._t_GRB = duration_of_GRB
        self._ra = ra
        self._dec = dec
        self._psd_eff = psd_eff

    def _spectrum_bins(self, spec):
        """
        Calculate the spectrum in the defined energy bins
        :param spec: a spectrum function spec(E)
        :return: spectrum binned in the defined energy bins
        """
        e2 = self.ene_max
        e1 = self.ene_min 
        return (e2 - e1) / 6.0 * (spec(e1) +
                                  4 * spec((e1 + e2) / 2.0) +
                                 spec(e2))

        
    def add_sources_simulated(self):

        assert self._shapes is not None, 'Please give at least one function for spectrum of source'

        
        first_bin = np.argmax(self.time_bins_start[self.time_bins_start<0])
        index_range = first_bin+1 + range(len(self.time_bins_start[np.logical_and(self.time_bins_start>0, self.time_bins_start<self._t_GRB)]))      
        wgt_time = np.array([])
        #np.random.seed(1000)
        for i in index_range:
            if self.time_bins_start[i]<self._t_GRB/3.:
                wgt_time = np.append(wgt_time, ((self.time_bins_start[i])/(self._t_GRB/3.))**(2.))
            else:
                wgt_time = np.append(wgt_time, ((self._t_GRB/3.)/(self.time_bins_start[i]))**(2.))
        wgt_time =np.ones_like(wgt_time)

        
        if "single" in self._event_types:
            for d in self.energy_and_time_bin_sgl_dict.keys():
                self._response_object.set_location(self._ra, self._dec)
                eff_area = self._response_object.get_response_det(d)

                for s in self._shapes.keys():
                    Flux = self._spectrum_bins(self._shapes[s])
                    for i in index_range:
                        if len(eff_area.shape)==1:
                            self.energy_and_time_bin_sgl_dict[d][i] += np.random.poisson((1-self._psd_eff)*eff_area*Flux*wgt_time[i-index_range[0]]*self.time_bin_length[i])
                            self.energy_and_time_bin_psd_dict[d][i] += np.random.poisson((self._psd_eff)*eff_area*Flux*wgt_time[i-index_range[0]]*self.time_bin_length[i])

                        else:
                            self.energy_and_time_bin_sgl_dict[d][i] += np.random.poisson((1-self._psd_eff)*np.dot(eff_area,Flux*wgt_time[i-index_range[0]])*self.time_bin_length[i])
                            self.energy_and_time_bin_psd_dict[d][i] += np.random.poisson((self._psd_eff)*np.dot(eff_area,Flux*wgt_time[i-index_range[0]])*self.time_bin_length[i])
                
                
        if "double" in self._event_types: 
            for d in self.energy_and_time_bin_me2_dict.keys():
                self._response_object.set_location(self._ra, self._dec)
                eff_area = self._response_object.get_response_det(d)

                for i in index_range:
                    if len(eff_area.shape)==1:
                        self.energy_and_time_bin_me2_dict[d][i] += np.random.poisson(eff_area*Flux_max*wgt_time[i-index_range[0]]*self.time_bin_length[i])
                        
                    else:
                        self.energy_and_time_bin_me2_dict[d][i] += np.random.poisson(np.dot(eff_area,Flux_max*wgt_time[i-index_range[0]])*self.time_bin_length[i])
                        
        if "triple" in self._event_types: 
            for d in self.energy_and_time_bin_me3_dict.keys():
                self._response_object.set_location(self._ra, self._dec)
                eff_area = self._response_object.get_response_det(d)

                for i in index_range:
                    if len(eff_area.shape)==1:
                        self.energy_and_time_bin_me3_dict[d][i] += np.random.poisson(eff_area*Flux_max*wgt_time[i-index_range[0]]*self.time_bin_length[i])
                    else:
                        self.energy_and_time_bin_me3_dict[d][i] += np.random.poisson(np.dot(eff_area,Flux_max*wgt_time[i-index_range[0]])*self.time_bin_length[i])
                        
        print('Added GRB from 0 to {} seconds at position ra {} dec {} to binned sgl data'.format(self._t_GRB, self._ra, self._dec))

        
