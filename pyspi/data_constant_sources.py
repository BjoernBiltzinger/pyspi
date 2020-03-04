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

class DataConstantSources(object):

    def __init__(self, pointings_list=None, position=None, event_types=["single"], afs=True, ebounds=None):
        """
        Init the data object for constant source analysis (no transients). In this case
        we can sum all the data of one pointing together!
        
        :param pointings_list: List of all pointings that shoud be used in the analysis
        :param position: Certain position in J2000 ([ra, dec]); All pointings which had 
        this position in the FOV will be used
        :param event_types: Which event types should be used?
        :param afs: Use afs data access?
        :param ebounds: Energy boundaries for analysis
        """

        assert (pointings_list is None) or (position is None), \
            "It is not possible to give a pointing_list and a position. Please choose one."

        assert (pointings_list is not None) or (position is not None), \
            "Please give a wanted pointings_list or a position in J2000"

        self._event_types = event_types

        self._afs = afs
        
        if position is not None:
            pointings_list = self._get_pointings_list(position)

        self._pointings_list = [str(i) for i in pointings_list]

        self._get_data_files()
        
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

        self._get_pointings_data()

        self._get_good_sgl_dets()
        
    def _get_data_files(self):
        """
        Get all the needed data files.

        :return:
        """
            
        # Download the data of all wanted pointings
        print('I will download now the data of {} pointings (if it was not already downloaded in an earlier run). In total I have to download about {} MB.'.format(len(self._pointings_list), len(self._pointings_list)*10))

        with progress_bar(len(self._pointings_list), title='Checking if data files are cached. If not they are downloaded.') as p:

            for pointing in self._pointings_list:
                if self._afs:
                    try:
                        # Get the data from the afs server
                        get_files_afs(pointing)
                    except:
                        # Get the files from the iSDC data archive
                        print('AFS data access did not work. I will try the ISDC data archive.')
                        get_files_isdcarc(pointing)
                else:
                    # Get the files from the iSDC data archive 
                    get_files_isdcarc(pointing)
                    
                p.increase()

            
    def _get_pointings_list(self, position):
        """
        Get the list of pointings that had the position in the FOV
        :param position: Position in J2000 ([ra, dec])
        :return: List of pointings that contained position in FOV
        """
        pass

    def _get_pointings_data(self):
        """
        Gets the data of all pointings (summed counts per energy bin, detector and pointing) 
        and exposure of all pointings.
        :return:
        """
        if "single" in self._event_types:
            counts_sgl = np.zeros((len(self._pointings_list), 19, len(self._ene_min)))
            #exposure_sgl = np.zeros((len(self._pointings_list), 19))

            counts_psd = np.zeros((len(self._pointings_list), 19, len(self._ene_min)))
            #exposure_psd = np.zeros((len(self._pointings_list), 19))            
            
        if "double" in self._event_types:
            counts_me2 = np.zeros((len(self._pointings_list), 42, len(self._ene_min)))
            #exposure_me2 = np.zeros((len(self._pointings_list), 42))

        if "triple" in self._event_types:
            counts_me3 = np.zeros((len(self._pointings_list), 24, len(self._ene_min)))
            #exposure_me3 = np.zeros((len(self._pointings_list), 24))

        exposure = np.zeros(len(self._pointings_list))

        start_times = []
        
        with progress_bar(len(self._pointings_list), title='Bin in energy and sum together for all pointings...') as p:

            for h, pointing in enumerate(self._pointings_list):
                with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing, 'spi_oper.fits.gz')) as hdu_oper:

                    start_times.append(Time(hdu_oper[1].header['TSTART']+51544, format='mjd', scale='tt'))
                    # Energy, time and dets of all events
                    if "single" in self._event_types:
                        energy_sgl = hdu_oper[1].data['energy']
                        dets_sgl = hdu_oper[1].data['DETE']

                        energy_psd = hdu_oper[2].data['energy']
                        dets_psd = hdu_oper[2].data['DETE']

                    if "double" in self._event_types:
                        energy_me2 = hdu_oper[4].data['energy']
                        dets_me2 = hdu_oper[4].data['DETE']

                    if "triple" in self._event_types:
                        energy_me3 = hdu_oper[5].data['energy']
                        dets_me3 = hdu_oper[5].data['DETE']

                    exposure[h] = hdu_oper[1].header['TELAPSE']
                # Build dic with entry for every det (0-84)
                # For sgl and psd only one det is hit

                if "single" in self._event_types:
                    sgl_energy_dict = {}
                    for i in range(19):
                        mask = dets_sgl==i
                        #if True in mask:
                        sgl_energy_dict[i] = energy_sgl[mask]
                    counts_sgl[h,:,:] = self._energy_bin_data(sgl_energy_dict)

                    # Get a list with the dets that seem to be defect (because there are 0 counts in them)
                    bad_sgl_dets = np.zeros(19, dtype=bool)
                    for i in range(19):
                        if sgl_energy_dict[i].size==0:
                            bad_sgl_dets[i] = True

                # PSD events
                    psd_energy_dict = {}
                    for i in range(19):
                        mask = dets_psd==i
                        psd_energy_dict[i] = energy_psd[mask]
                    counts_psd[h,:,:] = self._energy_bin_data(psd_energy_dict)
                    # Get a list with the dets that seem to be defect (because there are 0 counts in them)
                    bad_psd_dets = np.zeros(19, dtype=bool)
                    for i in range(19):
                        if psd_energy_dict[i].size==0:
                            bad_psd_dets[i] = True

                # Double events
                if "double" in self._event_types:
                    me2_energy_dict = {}
                    # For me2 events two dets are hit
                    for n, (i, k) in enumerate(double_names.values(), start=19):
                            mask1 = np.logical_and(dets_me2[:,0]==i, dets_me2[:,1]==k)
                            mask2 = np.logical_and(dets_me2[:,0]==k, dets_me2[:,1]==i)

                            e_array1 = energy_me2[mask1]
                            e_array2 = energy_me2[mask2]

                            total_e_array = np.concatenate((e_array1, e_array2))

                            me2_energy_dict[n] = np.sum(total_e_array, axis=1)

                    counts_me2[h,:,:] = self._energy_bin_data(me2_energy_dict)
                    # Get a list with the dets that seem to be defect (because there are 0 counts in them)
                    bad_me2_dets = np.zeros(42, dtype=bool)
                    for i in range(19,61):
                        if me2_energy_dict[i].size==0:
                            bad_me2_dets[i-19] = True

                # Triple events
                if "triple" in self._event_types:
                    me3_energy_dict = {}
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

                        total_e_array = np.concatenate((e_array1, e_array2, e_array3, e_array4, e_array5, e_array6))
                        me3_energy_dict[n] = np.sum(total_e_array, axis=1)

                    counts_me3[h,:,:] = self._energy_bin_data(me3_energy_dict)

                    # Get a list with the dets that seem to be defect (because there are 0 counts in them)
                    bad_me3_dets = np.zeros(24, dtype=bool)
                    for i in range(61,85):
                        if me3_energy_dict[i].size==0:
                            bad_me3_dets[i-61] = True
                p.increase()
        if "single" in self._event_types:
            self._counts_sgl = counts_sgl
            self._counts_psd = counts_psd
            
        if "double" in self._event_types:
            self._counts_me2 = counts_me2
            
        if "triple" in self._event_types:
            self._counts_me3 = counts_me3

        self._exposure = exposure

        self._start_times = start_times

    def _energy_bin_data(self, energy_tagged_data):
        """
        Function to bin the data in user defined energy bins
        :param energy_tagged_data: Dict with energy tagged 
        :return: Energy binned data
        """
        
        energy_bin = np.zeros((len(energy_tagged_data), len(self._ene_min)))

        for j, d in enumerate(energy_tagged_data.keys()):
        
            for i in range(len(self._ene_min)):

                energy_bin[j,i] = np.sum(np.where(np.logical_and(energy_tagged_data[d]>=
                                                                 self._ene_min[i],
                                                                 energy_tagged_data[d]<
                                                                 self._ene_max[i]),
                                                  1, 0))
                
        return energy_bin

    def plot_data(self, det):
        """
        Plot the data for all pointings for one det
        :param: For which detector?
        :return: figure
        """

        assert det in range(0,85), 'Please use a valid detector id between 0 and 84'

        if det in range(0,19):
            assert "single" in self._event_types, 'Single dets not used in this analysis.'
            counts_needed = self._counts_sgl[:,det,:]
            counts_psd_needed = self._counts_psd[:,det,:]
            counts_sum_needed = counts_psd_needed+counts_needed
            
        if det in range(19,19+42):
            assert "double" in self._event_types, 'Double dets not used in this analysis.'
            counts_needed = self._counts_me2[:,det-19,:]
            
        if det in range(19+42,19+42+24):
            assert "triple" in self._event_types, 'Triple dets not used in this analysis.'
            counts_needed = self._counts_me3[:,det-19-42,:]
            
        if len(self._pointings_list)==1:
            n_column =1
            n_row = 1
        elif len(self._pointings_list)==2:
            n_column =2
            n_row = 1
        elif len(self._pointings_list)<=4:
            n_column =2
            n_row = 2
        else:
            n_column = 3
            n_row = np.ceil(len(self._pointings_list)/3.)

        fig = plt.figure()
        for i, pointing in enumerate(self._pointings_list):

            ax = fig.add_subplot(n_row, n_column, i+1)
            ax.step(self._ebounds[1:], counts_needed[i], where='post', label='Non PSD')
            if det in range(0,19):
                ax.step(self._ebounds[1:], counts_psd_needed[i], where='post', label='PSD')
                ax.step(self._ebounds[1:], counts_sum_needed[i], where='post', label='Sum')
                
            ax.legend()
            ax.set_ylabel('Count Rates [cnts $s^{-1}$]')
            ax.set_xlabel('Energy [kev]')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(pointing)

    def _get_good_sgl_dets(self):
        """
        Get a good sgl dets for all pointings sep (the ones with more than 0 counts)
        :return:
        """
        good_sgl_dets = []
        for n in range(len(self._pointings_list)):
            good_sgl_dets_pointing = []
            for i in range(19):
                total_sum_det = np.sum(self.counts_sgl_with_psd[n,i])
                if total_sum_det!=0:
                    good_sgl_dets_pointing.append(i)
            good_sgl_dets.append(good_sgl_dets_pointing)
        self._good_sgl_dets = good_sgl_dets

    @property
    def start_times(self):
        return self._start_times
        
    @property
    def good_sgl_dets(self):
        return self._good_sgl_dets
    
    @property
    def counts_sgl_with_psd(self):
        return self._counts_sgl + self._counts_psd

    @property
    def counts_psd(self):
        return self._counts_psd

    @property
    def counts_me2(self):
        return self._counts_me2

    @property
    def counts_me3(self):
        return self._counts_me3

    @property
    def geometry_file_paths(self):
        geom_file_path_list = [os.path.join(get_path_of_external_data_dir(),
                                            'pointing_data', i,
                                            'sc_orbit_param.fits.gz')
                                            for i in self._pointings_list]

        return geom_file_path_list

    @property
    def ene_max(self):
        return self._ene_max

    @property
    def ene_min(self):
        return self._ene_min

    @property
    def ebounds(self):
        return self._ebounds
