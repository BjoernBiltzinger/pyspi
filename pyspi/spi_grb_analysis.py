from pyspi.spi_data import *
from pyspi.spi_response import ResponsePhotopeak, ResponseRMF
from pyspi.spi_pointing import *
from pyspi.spi_frame import *
from pyspi.utils.likelihood import Likelihood

from astromodels import *
import astropy.units as u
from astropy.coordinates import ICRS, Galactic, SkyCoord
from astropy.time.core import Time
from datetime import datetime
import os
from shutil import copyfile, rmtree
import sys

from pyspi.io.package_data import get_path_of_external_data_dir

from threeML.utils.statistics.likelihood_functions import *
from threeML import *

import seaborn as sns
sns.set_palette('pastel')

class GRBAnalysis(object):

    def __init__(self, configuration, likelihood_model):
        """
        Init a Spi Analysis object for an analysis of a GRB.
        :param configuration: Configuration dictionary
        :param likelihood_model: The inital astromodels likelihood_model
        """

        # TODO: Add a test if the configuration file is valid for a GRB analysis
        # Which event types should be used?
        self._event_types = configuration['Event_types']

        # Which energy range?
        self._emin = float(configuration['emin'])
        self._emax = float(configuration['emax'])

        # Binned or unbinned analysis?
        self._binned = configuration['Energy_binned']
        if self._binned:
            # Set ebounds of energy bins
            self._ebounds = np.array(configuration['Ebounds'])
            # If no ebounds are given use the default ones
            if self._ebounds is None:
                self._ebounds = np.logspace(np.log10(self._emin), np.log10(self._emax), 30)
            # Construct final energy bins if single events are used
            if 'single' in self._event_types:
                self._construct_energy_bins()
        else:
            raise NotImplementedError('Unbinned analysis not implemented!')

        # We need one nuisance parametert if we use the single dets for the efficiency
        # of the PSD detections in the electronic noise range. Init it to 0.85. Will be changed
        # by SPILike class if the single events are used
        if "single" in self._event_types:
            self._eff_psd = 0.85
        
        # Time of GRB. Needed to get the correct pointing.
        time_of_grb = configuration['Time_of_GRB_UTC']
        time = datetime.strptime(time_of_grb, '%y%m%d %H%M%S')
        self._time_of_grb = Time(time)

        # Active_Time of GRB 'start-stop' format
        self._active_time = configuration['Active_Time']

        # Which bkg estimation should be used (polynominal or background model)
        self._bkg_estimation = configuration['Bkg_estimation']

        if self._bkg_estimation=='Polynominal':    
            self._bkg_time_1 = configuration['Background_time_interval_1']
            self._bkg_time_2 = configuration['Background_time_interval_2']
        else:
            raise NotImplementedError('Background model is not yet implemented.')
            
        # Simmulate a GRB at the given time? Only used for testing!
        self._simulate = configuration['Simulate']

        if self._simulate is not None:
            
            sys.stdout.write('CAUTION! You selected to simmulate a source at the given active time! If you want to analyse a real GRB this will lead to completly wrong results! Please confirm that you wanted to simmulate a source [y/n]. ')
            
            # raw_input returns the empty string for "enter"
            yes = {'yes','y', 'ye', ''}
            no = {'no','n'}

            choice = raw_input().lower()
            if choice in yes:
               pass
            elif choice in no:
               raise AssertionError("Okay. I will aboard the calculation here. Please set the 'Simmulate' flag in the config file to False")
            else:
               sys.stdout.write("Please respond with 'y' or 'n'")
            self._simulate_dict = {}

            n_source = 0
            for s in self._simulate.keys():
                if s=="General":
                    self._simulate_ra = self._simulate[s]['ra']
                    self._simulate_dec = self._simulate[s]['dec']
                    self._simulate_duration = self._simulate[s]['duration']
                    self._simulate_response = self._simulate[s]['simulate_response']

                else:
                    assert "Function" in self._simulate[s].keys(), "I need the Function info to build the source!"
                
                    func = self._simulate[s]["Function"]
                    if func == 'Band':
                        shape = Band()
                        assert "K" in self._simulate[s].keys(), "Please give a normalization for the Band-function (K)."
                        assert "alpha" in self._simulate[s].keys(), "Please give a low energy slope for the Band-function (alpha)."
                        assert "beta" in self._simulate[s].keys(), "Please give a high energy slope for the Band-function (beta)"
                        assert "xp" in self._simulate[s].keys(), "Please give a peak energy for the Band-function (xp)"

                        shape.K = self._simulate[s]["K"]
                        shape.alpha = self._simulate[s]["alpha"]
                        shape.beta = self._simulate[s]["beta"]
                        shape.xp = self._simulate[s]["xp"]

                    elif func == 'Pl':
                        shape = Powerlaw()
                        assert "K" in self._simulate[s].keys(), "Please give a normalization for the Powerlaw-function (K)."
                        assert "index" in self._simulate[s].keys(), "Please give a slope for the Powerlaw-function (index)."

                        shape.K = self._simulate[s]["K"]
                        shape.index = self._simulate[s]["index"]

                    elif func == 'Cpl':
                        shape = Cutoff_powerlaw()
                        assert "K" in self._simulate[s].keys(), "Please give a normalization for the Cutoff-Powerlaw-Function (K)."
                        assert "index" in self._simulate[s].keys(), "Please give a slope for the Cutoff-Powerlaw-function (index)."
                        assert "xc" in self._simulate[s].keys(), "Please give a cutoff energy for the Cutoff-Powerlaw-function (xc)"

                        shape.K = self._simulate[s]["K"]
                        shape.index = self._simulate[s]["index"]
                        shape.xc = self._simulate[s]["xc"]

                    elif func == 'Line':
                        shape = Gaussian_Line()
                        assert "K" in self._simulate[s].keys(), "Please give a normalization for the Gaussian-Line-function (K)."
                        assert "E0" in self._simulate[s].keys(), "Please give an energy for the Gaussian-Line--function (E0)."
                        assert "sigma" in self._simulate[s].keys(), "Please give a width for the Gaussian-Line-function (sigma)"

                        shape.K = self._simulate[s]["K"]
                        shape.E0 = self._simulate[s]["E0"]
                        shape.sigma = self._simulate[s]["sigma"]
                    else:
                        raise NotImplementedError('Only Band, Pl, Cpl and Line are implemented')

                    self._simulate_dict[n_source] = shape
                    n_source += 1
                
        # Translate the input time ranges in start and stop times
        self._get_start_and_stop_times()

        # Get the unique name for the analysis (this name will be used to save the results later)
        self._analysis_name = configuration['Unique_analysis_name']

        # Check if the unique name is given and if it was never used before. If not add a time stamp to it to avoid
        # overwritting of old results
        current_time = datetime.now()
        if self._analysis_name==None:
            
            self._analysis_name = 'fit_{}'.format(current_time.strftime("%m-%d_%H-%M"))

            print('You have set no unique name for this analysis. \
                   I will name it according to the present time {}'.format(self._analysis_name))
            
        elif self._analysis_name in os.listdir(os.join.path(get_path_of_external_data_dir(),
                                                            self._analysis)):
            
            self._analysis_name = '{}_{}'.format(self._analysis_name,
                                                 current_time.strftime("%m-%d_%H-%M"))
            
            print('You have set a unique name for this analysis that was already used before. \
                   I will add the present time to the name {}'.format(self._analysis_name))
            
        # Init the response
        self._init_response()
        
        # Init the data
        self._init_data()

        
        # Which dets should be used? Can also be 'All' to use all possible
        # dets of the specified event_types
        self._dets = np.array(configuration['Detectors_to_use'])

        # Construct final det selection
        self._det_selection()
        
        # Get the background estimation
        self.get_background_estimation()

        # Init the SPI frame (to transform J2000 to SPI coordinates)
        self._init_frame()

        # Set the model 
        self.set_model(likelihood_model)

        # Set Likelihood class
        self._init_likelihood()
        
    def set_psd_eff(self, value):
        """
        Set the psd efficency
        :param value: Value for psd eff
        :return:
        """
        self._eff_psd = value

    def _construct_energy_bins(self):
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
        # Case 1400-17000 is completly in the ebound range
        if self._ebounds[0]<psd_low_energy and self._ebounds[-1]>psd_high_energy:
            psd_bin = True
            start_found = False
            stop_found = False
            for i, e in enumerate(self._ebounds):
                if e>=psd_low_energy and not start_found:
                    start_index = i
                    start_found=True
                if e>=1700 and not stop_found:
                    stop_index = i
                    stop_found=True
            self._ebounds = np.insert(self._ebounds, start_index, psd_low_energy)
            self._ebounds = np.insert(self._ebounds, stop_index+1, psd_high_energy)

            if stop_index-start_index>1:
                sgl_mask = np.logical_and(np.logical_or(self._ebounds[:-1]<=psd_low_energy,
                                                        self._ebounds[:-1]>=psd_high_energy),
                                          np.logical_or(self._ebounds[1:]<=psd_low_energy,
                                                        self._ebounds[1:]>=psd_high_energy))
            elif stop_index-start_index==1:
                sgl_mask = np.ones(len(self._ebounds)-1, dtype=bool)
                sgl_mask[start_index] = False
                sgl_mask[stop_index] = False
            elif stop_index-start_index==0:
                sgl_mask = np.ones(len(self._ebounds)-1, dtype=bool)
                sgl_mask[start_index] = False
            change = True
        # Upper bound of erange in psd bin
        elif self._ebounds[0]<psd_low_energy and self._ebounds[-1]>psd_low_energy:
            psd_bin = True
            start_found = False
            for i, e in enumerate(a):
                if e>=psd_low_energy and not start_found:
                    start_index = i
                    start_found=True
            self._ebounds = np.insert(self._ebounds, start_index, psd_low_energy)
            sgl_mask = (self._ebounds<psd_low_energy)[:-1]
            change = True
        # Lower bound of erange in psd bin
        elif self._ebounds[0]<psd_high_energy and self._ebounds[-1]>psd_high_energy:
            psd_bin = True
            stop_found = False
            for i, e in enumerate(a):
                if e>=psd_high_energy and not stop_found:
                    stop_index = i
                    stop_found=True
            a = np.insert(self._ebounds, stop_index, psd_high_energy)
            sgl_mask = (self._ebounds>=psd_high_energy)[:-1]
            change=True
        # else erange completly outside of psd bin => all just single
        else:
            sgl_mask = np.ones_like(self._ebounds[:-1], dtype=bool)

        self._sgl_mask = sgl_mask

        if change:
            print('I had to readjust the ebins to avoid having ebins inside of the single event electronic noise energy range. The new boundaries of the ebins are: {}.'.format(self._ebounds))
        
    def _det_selection(self):
        """
        Function to figure out which dets should be used in the analysis. Takes user input and 
        not working detectors into account.
        :return:
        """
        if self._dets=="All":

            # If all dets should be used we will just ignore the ones that are turned off
            if "single" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                bad_sgl_dets = self._data_object.bad_sgl_dets
                
                self._sgl_dets_to_use = np.arange(0,19,1,dtype=int)[~bad_sgl_dets]

            # If all dets should be used we will just ignore the ones that are turned off
            #if "psd" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                #bad_psd_dets = self._data_object.bad_psd_dets
                
                #self._psd_dets_to_use = np.arange(0,19,1,dtype=int)[~bad_sgl_dets]
                
            if "double" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                bad_me2_dets = self._data_object.bad_me2_dets
                
                self._me2_dets_to_use = np.arange(19,61,1,dtype=int)[~bad_me2_dets]
                
            if "triple" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                bad_me3_dets = self._data_object.bad_me3_dets

                self._me3_dets_to_use = np.arange(61,85,1,dtype=int)[~bad_me3_dets]
            
        else:

            # Check if all input dets are valid detector ids
            for d in self._dets:
                assert d in np.arange(85), "{} is not a valid detector. Please only use detector ids between 0 and 84.".format(d)

            # Sort the input detector ids in single, double and triple
            single_dets = self._dets[self._dets<19]
            double_dets = self._dets[np.logical_and(self._dets>=19, self._dets<61)]
            triple_dets = self._dets[self._dets>=61]

            # If single event type is wanted build the array which contains the dets
            if "single" in self._event_types:
                bad_sgl_dets = self._data_object.bad_sgl_dets
                self._sgl_dets_to_use = np.array([], dtype=int)
                for s in single_dets:
                    if not bad_sgl_dets[s]:
                        self._sgl_dets_to_use = np.append(self._sgl_dets_to_use, s)
                    else:
                        warnings.warn("You wanted to use detector {}, but it is turned off. I will ignore this detector for the rest of the analysis.".format(s))
                assert self._sgl_dets_to_use.size>0, 'All the detectors you want to use are turned off for this pointing...' 
            else:
                if single_dets.size>0:
                    warnings.warn("You wanted to use some single dets but did not select the single detection event type. I will ignore all single_dets for the rest of the calculation. If you want to use them please restart the calculation with single in the event_types list of the configuration file.")

            if "double" in self._event_types:
                bad_me2_dets = self._data_object.bad_me2_dets
                self._me2_dets_to_use = np.array([], dtype=int)
                for s in double_dets:
                    if not bad_me2_dets[s]:
                        self._me2_dets_to_use = np.append(self._me2_dets_to_use, s)
                    else:
                        warnings.warn("You wanted to use detector {}, but it is turned off. I will ignore this detector for the rest of the analysis.".format(s))
                assert self._me2_dets_to_use.size>0, 'All the detectors you want to use are turned off for this pointing...' 
            else:
                if double_dets.size>0:
                    warnings.warn("You wanted to use some double dets but did not select the double detection event type. I will ignore all double_dets for the rest of the calculation. If you want to use them please restart the calculation with double in the event_types list of the configuration file.")
                
            if "triple" in self._event_types or triple_dets.size>0:
                bad_me3_dets = self._data_object.bad_me3_dets
                self._me3_dets_to_use = np.array([], dtype=int)
                for s in triple_dets:
                    if not bad_me3_dets[s]:
                        self._me3_dets_to_use = np.append(self._me3_dets_to_use, s)
                    else:
                        warnings.warn("You wanted to use detector {}, but it is turned off. I will ignore this detector for the rest of the analysis.".format(s))
                assert self._me3_dets_to_use.size>0, 'All the detectors you want to use are turned off for this pointing...' 
            else:
                if triple_dets.size>0:
                    warnings.warn("You wanted to use some triple dets but did not select the triple detection event type. I will ignore all triple_dets for the rest of the calculation. If you want to use them please restart the calculation with triple in the event_types list of the configuration file.")
                
    def _get_start_and_stop_times(self):
        """
        Get the input start and stop times of active time and bkg times
        :return:
        """
        # Get start and stop time of active time from 'start-stop' format
        split = self._active_time.split('-')
        if len(split)==4:
            self._active_start = -float(split[1])
            self._active_stop = -float(split[-1])
        elif len(split)==3:
            self._active_start = -float(split[1])
            self._active_stop = float(split[-1])
        else:
            self._active_start = float(split[0])
            self._active_stop = float(split[1])

        if self._bkg_estimation=="Polynominal":
            # Get start and stop time of active time from 'start-stop' format
            split = self._bkg_time_1.split('-')
            if len(split)==4:
                self._bkg1_start = -float(split[1])
                self._bkg1_stop = -float(split[-1])
            elif len(split)==3:
                self._bkg1_start = -float(split[1])
                self._bkg1_stop = float(split[-1])
            else:
                self._bkg1_start = float(split[0])
                self._bkg1_stop = float(split[1])

            # Get start and stop time of active time from 'start-stop' format
            split = self._bkg_time_2.split('-')
            if len(split)==4:
                self._bkg2_start = -float(split[1])
                self._bkg2_stop = -float(split[-1])
            elif len(split)==3:
                self._bkg2_start = -float(split[1])
                self._bkg2_stop = float(split[-1])
            else:
                self._bkg2_start = float(split[0])
                self._bkg2_stop = float(split[1])


    def _init_likelihood(self):
        """
        Initalize Likelihood object. Will be used later to calculate the sum of all log-likelihoods
        :return:
        """
        self._likelihood = Likelihood(numba_cpu=True, parallel=True)
        
    def _init_frame(self):
        """
        Initalize the spi frame object, that is needed to transform ICRS coordinates to
        spacecraft coordinates.
        :return:
        """
        self._pointing_object = SPIPointing(self._data_object.geometry_file_path)

        self._frame_object = SPIFrame(**self._pointing_object.sc_points[0])

        # get skycoord object of center ra and dec in icrs frame
        pointing_sat = SkyCoord(lon=0, lat=0, unit='deg', frame=self._frame_object)

        self._pointing_icrs = pointing_sat.transform_to('icrs')
        
    def _init_data(self):
        """
        Get the data object with all the data we need
        :return:
        """
        
        if self._simulate is None:
            self._data_object = DataGRB(self._time_of_grb, afs=True, ebounds=self._ebounds, event_types=self._event_types)

            if self._binned:
                # Bin the data in energy and time - dummy values for time bin step
                # size and ebounds
                self._data_object.time_and_energy_bin(time_bin_step=1.,
                                                          start=self._bkg1_start-10,
                                                          stop=self._bkg2_stop+10)
            else:
                raise NotImplementedError('Only binned analysis implemented at the moment!')
            
        else:
            #simulate_pos_icrs = SkyCoord(ra=self._simulate_ra, dec=self._simulate_dec,
            #                             unit='deg', frame='icrs')
            #print(simulate_pos_icrs)
            #simulate_ra_sat = simulate_pos_icrs.transform_to(self._frame_object).lon.deg
            #simulate_dec_sat = simulate_pos_icrs.transform_to(self._frame_object).lat.deg
            if self._simulate_response=='RMF':
                response_object_sim = ResponseRMF(ebounds=self._ebounds, time=self._time_of_grb)
            elif self._simulate_response=='Photopeak':
                response_object_sim = ResponsePhotopeak(ebounds=self._ebounds, time=self._time_of_grb)
            else:
                raise AssertionError
            

            self._data_object = DataGRBSimulate(self._time_of_grb, response_object_sim,
                                                    ra=self._simulate_ra, dec=self._simulate_dec,
                                                    duration_of_GRB=self._simulate_duration,
                                                    shapes = self._simulate_dict,
                                                    afs=True, ebounds=self._ebounds,
                                                    event_types=self._event_types)
            
            if self._binned:
                self._data_object.time_and_energy_bin(time_bin_step=1.,
                                                      start=self._bkg1_start-10,
                                                      stop=self._bkg2_stop+10)
                self._data_object.add_sources_simulated()
            else:
                raise NotImplementedError('Only binned analysis implemented at the moment!')
        # Build a mask to cover the time bins of the active time
        time_bins = self._data_object.time_bins
        
        self._active_time_mask = np.logical_and(time_bins[:,0]>self._active_start,
                                                time_bins[:,1]<self._active_stop)
    
        # Get time bins
        self._active_time_bins = self._data_object.time_bins[self._active_time_mask]

        self._real_start_active = self._active_time_bins[0,0]
        self._real_stop_active = self._active_time_bins[-1,-1]
        self._active_time_seconds = self._real_stop_active - self._real_start_active
            
        if 'single' in self._event_types:
            sgl_dets = self._data_object.energy_sgl_dict.keys()

            self._active_time_counts_energy_sgl_dict = {}
            for d in sgl_dets:
                self._active_time_counts_energy_sgl_dict[d] = np.sum(self._data_object.energy_and_time_bin_sgl_dict[d][self._active_time_mask], axis=0)

        #if 'psd' in self._event_types:
            psd_dets = self._data_object.energy_psd_dict.keys()

            self._active_time_counts_energy_psd_dict = {}
            for d in psd_dets:
                self._active_time_counts_energy_psd_dict[d] = np.sum(self._data_object.energy_and_time_bin_psd_dict[d][self._active_time_mask], axis=0)

            self._active_time_counts_energy_sgl_psd_sum_dict = {}
            for d in sgl_dets:
                self._active_time_counts_energy_sgl_psd_sum_dict[d] = self._active_time_counts_energy_psd_dict[d] + self._active_time_counts_energy_sgl_dict[d]

        if 'double' in self._event_types:
            me2_dets = self._data_object.energy_me2_dict.keys()

            self._active_time_counts_energy_me2_dict = {}
            for d in me2_dets:
                self._active_time_counts_energy_me2_dict[d] = np.sum(self._data_object.energy_and_time_bin_me2_dict[d][self._active_time_mask], axis=0)
                
        if 'triple' in self._event_types:
            me3_dets = self._data_object.energy_me3_dict.keys()

            self._active_time_counts_energy_me3_dict = {}
            for d in me3_dets:
                self._active_time_counts_energy_me3_dict[d] = np.sum(self._data_object.energy_and_time_bin_me3_dict[d][self._active_time_mask], axis=0)

    def update_model(self, likelihood_model):
        """
        Update the model with the new source positions and spectra
        :param likelihood_model: The new astromodels likelihood_model
        :return:
        """

        for point_source in likelihood_model.point_sources.values():

            # If point source has no free parameters we don't need to
            # recalculate the influence every time
            if point_source.has_free_parameters():

                # Calculate influence of source on every det
                self.update_pointsource(point_source.name, point_source)

        for extended_source in likelihood_model.extended_sources.values():

            raise NotImplementedError('Extended sources not yet implemented!')
        
    def set_model(self, likelihood_model):
        """
        Set the model with the inital source positions and spectra
        :param likelihood_model: The astromodels likelihood_model
        :return:
        """
        self._point_sources = {}
        for point_source in likelihood_model.point_sources.values():

            # Calculate influence of source on every det
            self.create_pointsource(point_source.name, point_source)

        self._extended_sources = {}
        for extended_source in likelihood_model.extended_sources.values():
            
            raise NotImplementedError('Extended sources not yet implemented!')

    def create_pointsource(self, name, point_source):
        """
        Update the influence of a point source on all dets
        :param name: Name of point source
        :param point_source: Astromodel point source 
        :return:
        """

        assert name not in self._point_sources.keys(), \
            'Can not create the source {} twice!'.format(name)
        
        # ra and dec to sat coord
        icrscoord = SkyCoord(ra=point_source.position.ra.value,
                             dec=point_source.position.dec.value,
                             unit='deg',
                             frame='icrs')
        
        satcoord = icrscoord.transform_to(self._frame_object)

        ra_sat = satcoord.lon.deg
        dec_sat = satcoord.lat.deg 

        response_sgl = {}
        response_psd = {}
        response_me2 = {}
        response_me3 = {}
        
        self._response_object.set_location(ra_sat, dec_sat)
        
        if 'single' in self._event_types:
            for d in self._sgl_dets_to_use:
                response_sgl[d] = self._response_object.get_response_det(d)

        #if 'psd' in self._event_types:
            for d in self._sgl_dets_to_use:
                response_psd[d] = response_sgl[d]*self._eff_psd # TODO: Check response for PSD events

        if 'double' in self._event_types:
            for d in self._me2_dets_to_use:
                response_me2[d] = self._response_object.get_response_det(d)

        if 'triple' in self._event_types:
            for d in self._me3_dets_to_use:
                response_me3[d] = self._response_object.get_response_det(d)

        # Get current spectrum of source
        spectrum_bins = self.calculate_spectrum(point_source)
        
        predicted_count_rates_sgl = {}
        predicted_count_rates_psd = {}
        predicted_count_rates_me2 = {}
        predicted_count_rates_me3 = {}
        # Get the predicted count rates in all dets in all PHA bins (individual for all pointings later)
        if 'single' in self._event_types:
            for d in self._sgl_dets_to_use:
                predicted_count_rates_sgl[d] = self._fold(response_sgl[d], spectrum_bins)*self._active_time_seconds

        #if 'psd' in self._event_types:
            for d in self._sgl_dets_to_use:
                predicted_count_rates_psd[d] = self._fold(response_psd[d], spectrum_bins)*self._active_time_seconds

        if 'double' in self._event_types:
            for d in self._me2_dets_to_use:
                predicted_count_rates_me2[d] = self._fold(response_me2[d], spectrum_bins)*self._active_time_seconds

        if 'triple' in self._event_types:
            for d in self._me3_dets_to_use:
                predicted_count_rates_me3[d] = self._fold(response_me3[d], spectrum_bins)*self._active_time_seconds

        spectal_parameters = {}
        
        for i, component in enumerate(point_source._components.values()):

            spectral_parameters_component = {}
            for key in component.shape.parameters.keys():
                spectral_parameters_component[key] = component.shape.parameters[key].value

            spectal_parameters[i] = spectral_parameters_component
                
        # Create the entry of the point source
        self._point_sources[name] = {'ra': point_source.position.ra.value,
                                     'dec': point_source.position.dec.value,
                                     'response_sgl': response_sgl,
                                     'response_psd': response_psd,
                                     'response_me2': response_me2,
                                     'response_me3': response_me3,
                                     'predicted_count_rates_sgl': predicted_count_rates_sgl,
                                     'predicted_count_rates_psd': predicted_count_rates_psd,
                                     'predicted_count_rates_me2': predicted_count_rates_me2,
                                     'predicted_count_rates_me3': predicted_count_rates_me3,
                                     'spectal_parameters': spectal_parameters}

            
    def update_pointsource(self, name, point_source):
        """
        Update the influence of a point source on all dets
        :param name: Name of point source
        :param point_source: Astromodel point source 
        :return:
        """

        assert name in self._point_sources.keys(), \
            'The source with the name {} does not exists yet. We can not create a new source on the fly!'.format(name)
        
        # if position has changed recalculate response with new position - response is a callable function of Energy
        if point_source.position.ra.value != self._point_sources[name]['ra'] or \
           point_source.position.dec.value != self._point_sources[name]['dec']:

            # ra and dec to sat coord
            icrscoord = SkyCoord(ra=point_source.position.ra.value,
                                 dec=point_source.position.dec.value,
                                 unit='deg',
                                 frame='icrs')

            satcoord = icrscoord.transform_to(self._frame_object)

            ra_sat = satcoord.lon.deg
            dec_sat = satcoord.lat.deg

            response_sgl = {}
            response_psd = {}
            response_me2 = {}
            response_me3 = {}

            self._response_object.set_location(ra_sat,
                                               dec_sat)
            
            if 'single' in self._event_types:
                for d in self._sgl_dets_to_use:
                    response_sgl[d] = self._response_object.get_response_det(d)

            #if 'psd' in self._event_types:
                for d in self._sgl_dets_to_use:
                    response_psd[d] = response_sgl[d]*self._eff_psd #TODO: Check correct response for psd events

            if 'double' in self._event_types:
                for d in self._me2_dets_to_use:
                    response_me2[d] = self._response_object.get_response_det(d) 

            if 'triple' in self._event_types:
                for d in self._me3_dets_to_use:
                    response_me3[d] = self._response_object.get_response_det(d)
                    
        else:
            response_sgl = self._point_sources[name]['response_sgl']
            response_psd = {}
            for key in response_sgl:
                response_psd[key] = response_sgl[key]*self._eff_psd
            response_me2 = self._point_sources[name]['response_me2']
            response_me3 = self._point_sources[name]['response_me3']
            
        # Get current spectrum of source
        spectrum_bins = self.calculate_spectrum(point_source)

        predicted_count_rates_sgl = {}
        predicted_count_rates_psd = {}
        predicted_count_rates_me2 = {}
        predicted_count_rates_me3 = {}
        # Get the predicted count rates in all dets in all PHA bins (individual for all pointings later)
        if 'single' in self._event_types:
            for d in self._sgl_dets_to_use:
                predicted_count_rates_sgl[d] = self._fold(response_sgl[d], spectrum_bins)*self._active_time_seconds
            for d in self._sgl_dets_to_use:
                predicted_count_rates_psd[d] = self._fold(response_psd[d], spectrum_bins)*self._active_time_seconds
            
        if 'double' in self._event_types:
            for d in self._me2_dets_to_use:
                predicted_count_rates_me2[d] = self._fold(response_me2[d], spectrum_bins)*self._active_time_seconds

        if 'triple' in self._event_types:
            for d in self._me3_dets_to_use:
                predicted_count_rates_me3[d] = self._fold(response_me3[d], spectrum_bins)*self._active_time_seconds
                
        spectal_parameters = {}
        for i, component in enumerate(point_source._components.values()):

            spectral_parameters_component = {}
            for key in component.shape.parameters.keys():
                spectral_parameters_component[key] = component.shape.parameters[key].value

            spectal_parameters[i] = spectral_parameters_component

        # Update the entry of the point source
        self._point_sources[name] = {'ra': point_source.position.ra.value,
                                     'dec': point_source.position.dec.value,
                                     'response_sgl': response_sgl,
                                     'response_psd': response_psd,
                                     'response_me2': response_me2,
                                     'response_me3': response_me3,
                                     'predicted_count_rates_sgl': predicted_count_rates_sgl,
                                     'predicted_count_rates_psd': predicted_count_rates_psd,
                                     'predicted_count_rates_me2': predicted_count_rates_me2,
                                     'predicted_count_rates_me3': predicted_count_rates_me3,
                                     'spectal_parameters': spectal_parameters}

    def calculate_spectrum(self, spec):
        """
        Calculate the spectrum in the defined energy bins
        :param spec: a spectrum function spec(E)
        :return: spectrum binned in the defined energy bins
        """
        e2 = self._data_object.ene_max
        e1 = self._data_object.ene_min 
        return (e2 - e1) / 6.0 * (spec(e1) +
                                  4 * spec((e1 + e2) / 2.0) +
                                 spec(e2))
            
        
    def get_background_estimation(self):
        """
        Function to get the background estimation for the individual detectors
        - Polynominal for GRB analysis
        - Background model for constant point sources or extended_sources
        :return:
        """
        if 'single' in self._event_types:
            # Initalize the polynominal background fit to the time before and after the trigger time
            # to get the temporal changing Background
            self._bkg_sgl_psd_sum = {}
            self._bkg_active_sgl_psd_sum = {}


            for d in self._sgl_dets_to_use:

                # Use threeML times series to calculate the background polynominals for every det
                # and det PHA channel

                tsb = TimeSeriesBuilder.from_spi_spectrum('spi_det{}'.format(d),
                                                          self._data_object.time_bins_start,
                                                          self._data_object.time_bins_stop,
                                                          self._data_object.time_bin_length,
                                                          self._data_object.energy_and_time_bin_sgl_dict[d]+
                                                          self._data_object.energy_and_time_bin_psd_dict[d],
                                                          self._response_object,
                                                          poly_order=1)
                
                # Set active time and background time
                tsb.set_active_time_interval(self._active_time)
                tsb.set_background_interval(self._bkg_time_1, self._bkg_time_2)

                time_series = tsb.time_series
                polynomials = time_series.polynomials
                """
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._active_time_bins[:,0],
                                               self._active_time_bins[:,1])
                    for j, (start, stop) in enumerate(self._active_time_bins):

                        bkg_error[i,j] = p.integral_error(self._active_time_bins[j,0],
                                                          self._active_time_bins[j,1]) # Correct?

                """
                # General bkg counts and errors for time bins between self._bkg1_start and self._bkg2_stop
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._data_object.time_bins[:,0],
                                               self._data_object.time_bins[:,1])
                    for j, (start, stop) in enumerate(self._data_object.time_bins):

                        bkg_error[i,j] = p.integral_error(start,
                                                          stop)


                # Get the bkg counts and errors in the active time
                bkg_counts_active = np.empty((len(np.asarray(polynomials))))
                bkg_error_active = np.empty((len(np.asarray(polynomials))))
                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts_active[i] = p.integral(self._real_start_active, self._real_stop_active)

                    bkg_error_active[i] = p.integral_error(self._real_start_active,
                                                      self._real_stop_active) # Correct?
                #bkg_error_active = np.where(bkg_error_active==0, 1e-10, bkg_error_active)
                self._bkg_sgl_psd_sum[d] = {'error': bkg_error, 'counts': bkg_counts} 
                self._bkg_active_sgl_psd_sum[d] = {'error_active': bkg_error_active, 'counts_active': bkg_counts_active}  

        #if 'psd' in self._event_types:
            # Initalize the polynominal background fit to the time before and after the trigger time
            # to get the temporal changing Background
            self._bkg_psd = {}
            self._bkg_active_psd = {}


            for d in self._sgl_dets_to_use:

                # Use threeML times series to calculate the background polynominals for every det
                # and det PHA channel

                tsb = TimeSeriesBuilder.from_spi_spectrum('spi_det{}'.format(d),
                                                          self._data_object.time_bins_start,
                                                          self._data_object.time_bins_stop,
                                                          self._data_object.time_bin_length,
                                                          self._data_object.energy_and_time_bin_psd_dict[d],
                                                          self._response_object,
                                                          poly_order=1)
                
                # Set active time and background time
                tsb.set_active_time_interval(self._active_time)
                tsb.set_background_interval(self._bkg_time_1, self._bkg_time_2)

                time_series = tsb.time_series
                polynomials = time_series.polynomials
                """
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._active_time_bins[:,0],
                                               self._active_time_bins[:,1])
                    for j, (start, stop) in enumerate(self._active_time_bins):

                        bkg_error[i,j] = p.integral_error(self._active_time_bins[j,0],
                                                          self._active_time_bins[j,1]) # Correct?

                """
                # General bkg counts and errors for time bins between self._bkg1_start and self._bkg2_stop
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._data_object.time_bins[:,0],
                                               self._data_object.time_bins[:,1])
                    for j, (start, stop) in enumerate(self._data_object.time_bins):

                        bkg_error[i,j] = p.integral_error(start,
                                                          stop)


                # Get the bkg counts and errors in the active time
                bkg_counts_active = np.empty((len(np.asarray(polynomials))))
                bkg_error_active = np.empty((len(np.asarray(polynomials))))
                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts_active[i] = p.integral(self._real_start_active, self._real_stop_active)

                    bkg_error_active[i] = p.integral_error(self._real_start_active,
                                                      self._real_stop_active) # Correct?
                #bkg_error_active = np.where(bkg_error_active==0, 1e-10, bkg_error_active) 
                self._bkg_psd[d] = {'error': bkg_error, 'counts': bkg_counts} 
                self._bkg_active_psd[d] = {'error_active': bkg_error_active, 'counts_active': bkg_counts_active}  

        if 'double' in self._event_types:
            # Initalize the polynominal background fit to the time before and after the trigger time
            # to get the temporal changing Background
            self._bkg_me2 = {}
            self._bkg_active_me2 = {}


            for d in self._me2_dets_to_use:

                # Use threeML times series to calculate the background polynominals for every det
                # and det PHA channel
                tsb = TimeSeriesBuilder.from_spi_spectrum('spi_det{}'.format(d),
                                                          self._data_object.time_bins_start,
                                                          self._data_object.time_bins_stop,
                                                          self._data_object.time_bin_length,
                                                          self._data_object.energy_and_time_bin_me2_dict[d],
                                                          self._response_object,
                                                          poly_order=1)

                # Set active time and background time
                tsb.set_active_time_interval(self._active_time)
                tsb.set_background_interval(self._bkg_time_1, self._bkg_time_2)

                time_series = tsb.time_series
                polynomials = time_series.polynomials
                """
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._active_time_bins[:,0],
                                               self._active_time_bins[:,1])
                    for j, (start, stop) in enumerate(self._active_time_bins):

                        bkg_error[i,j] = p.integral_error(self._active_time_bins[j,0],
                                                          self._active_time_bins[j,1]) # Correct?

                """
                # General bkg counts and errors for time bins between self._bkg1_start and self._bkg2_stop
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._data_object.time_bins[:,0],
                                               self._data_object.time_bins[:,1])
                    for j, (start, stop) in enumerate(self._data_object.time_bins):

                        bkg_error[i,j] = p.integral_error(start,
                                                          stop)


                # Get the bkg counts and errors in the active time
                bkg_counts_active = np.empty((len(np.asarray(polynomials))))
                bkg_error_active = np.empty((len(np.asarray(polynomials))))
                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts_active[i] = p.integral(self._real_start_active, self._real_stop_active)

                    bkg_error_active[i] = p.integral_error(self._real_start_active,
                                                      self._real_stop_active) # Correct?
                #bkg_error_active = np.where(bkg_error_active==0, 1e-10, bkg_error_active) 
                self._bkg_me2[d] = {'error': bkg_error, 'counts': bkg_counts} 
                self._bkg_active_me2[d] = {'error_active': bkg_error_active, 'counts_active': bkg_counts_active}  

        if 'triple' in self._event_types:
            # Initalize the polynominal background fit to the time before and after the trigger time
            # to get the temporal changing Background
            self._bkg_me3 = {}
            self._bkg_active_me3 = {}


            for d in self._me3_dets_to_use:

                # Use threeML times series to calculate the background polynominals for every det
                # and det PHA channel
                tsb = TimeSeriesBuilder.from_spi_spectrum('spi_det{}'.format(d),
                                                          self._data_object.time_bins_start,
                                                          self._data_object.time_bins_stop,
                                                          self._data_object.time_bin_length,
                                                          self._data_object.energy_and_time_bin_me3_dict[d],
                                                          self._response_object,
                                                          poly_order=1)

                # Set active time and background time
                tsb.set_active_time_interval(self._active_time)
                tsb.set_background_interval(self._bkg_time_1, self._bkg_time_2)

                time_series = tsb.time_series
                polynomials = time_series.polynomials
                """
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._active_time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._active_time_bins[:,0],
                                               self._active_time_bins[:,1])
                    for j, (start, stop) in enumerate(self._active_time_bins):

                        bkg_error[i,j] = p.integral_error(self._active_time_bins[j,0],
                                                          self._active_time_bins[j,1]) # Correct?

                """
                # General bkg counts and errors for time bins between self._bkg1_start and self._bkg2_stop
                bkg_counts = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins)))
                bkg_error = np.empty((len(np.asarray(polynomials)),len(self._data_object.time_bins))) 

                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts[i] = p.integral(self._data_object.time_bins[:,0],
                                               self._data_object.time_bins[:,1])
                    for j, (start, stop) in enumerate(self._data_object.time_bins):

                        bkg_error[i,j] = p.integral_error(start,
                                                          stop)


                # Get the bkg counts and errors in the active time
                bkg_counts_active = np.empty((len(np.asarray(polynomials))))
                bkg_error_active = np.empty((len(np.asarray(polynomials))))
                for i, p in enumerate(np.asarray(polynomials)):
                    bkg_counts_active[i] = p.integral(self._real_start_active, self._real_stop_active)

                    bkg_error_active[i] = p.integral_error(self._real_start_active,
                                                      self._real_stop_active) # Correct?
                #bkg_error_active = np.where(bkg_error_active==0, 1e-10, bkg_error_active) 
                self._bkg_me3[d] = {'error': bkg_error, 'counts': bkg_counts} 
                self._bkg_active_me3[d] = {'error_active': bkg_error_active, 'counts_active': bkg_counts_active}  


    def _get_model_counts(self, likelihood_model):
        """
        Get the current model counts for all dets #TODO simplify this to only one possible source(This is a GRB analysis)
        :param likelihood_model: current likelihood_model
        :return:
        """
        
        self.update_model(likelihood_model)
        if "single" in self._event_types:
            expected_model_counts = np.zeros((len(self._sgl_dets_to_use),
                                              len(self._data_object.ene_min)))
            for i, d in enumerate(self._sgl_dets_to_use):
                for point_s in self._point_sources.keys():
                    expected_model_counts[i] += self._point_sources[point_s]['predicted_count_rates_sgl'][d]

            self._expected_model_counts_sgl_psd_sum = expected_model_counts

        #if "psd" in self._event_types:
            expected_model_counts = np.zeros((len(self._sgl_dets_to_use),
                                              len(self._data_object.ene_min)))
            for i, d in enumerate(self._sgl_dets_to_use):
                for point_s in self._point_sources.keys():
                    expected_model_counts[i] += self._point_sources[point_s]['predicted_count_rates_psd'][d]

            self._expected_model_counts_psd = expected_model_counts

        if "double" in self._event_types:
            expected_model_counts = np.zeros((len(self._me2_dets_to_use),
                                              len(self._data_object.ene_min)))
            for i, d in enumerate(self._me2_dets_to_use):
                for point_s in self._point_sources.keys():
                    expected_model_counts[i] += self._point_sources[point_s]['predicted_count_rates_me2'][d]

            self._expected_model_counts_me2 = expected_model_counts
            
        if "triple" in self._event_types:
            expected_model_counts = np.zeros((len(self._me3_dets_to_use),
                                              len(self._data_object.ene_min)))
            for i, d in enumerate(self._me3_dets_to_use):
                for point_s in self._point_sources.keys():
                    expected_model_counts[i] += self._point_sources[point_s]['predicted_count_rates_me3'][d]

            self._expected_model_counts_me3 = expected_model_counts
        
    def get_log_like(self, likelihood_model):
        """
        Return the current log likelihood value, only point_source at the moment
        :return: loglike value
        """
        
        self._get_model_counts(likelihood_model)

        # Use pgstat likelihood (background gaussian + signal poisson)
        loglike = 0
        if 'single' in self._event_types:
            for v, d in enumerate(self._sgl_dets_to_use):

                loglike += self._likelihood.PG_stat(
                    self._active_time_counts_energy_sgl_psd_sum_dict[d][self._sgl_mask],
                    self._bkg_active_sgl_psd_sum[d]['counts_active'][self._sgl_mask],
                    self._bkg_active_sgl_psd_sum[d]['error_active'][self._sgl_mask],
                    self._expected_model_counts_sgl_psd_sum[v][self._sgl_mask])
                
                #loglike += np.sum(poisson_observed_gaussian_background(
                #    self._active_time_counts_energy_sgl_psd_sum_dict[d][self._sgl_mask],
                #    self._bkg_active_sgl_psd_sum[d]['counts_active'][self._sgl_mask],
                #    self._bkg_active_sgl_psd_sum[d]['error_active'][self._sgl_mask],
                #    self._expected_model_counts_sgl_psd_sum[v][self._sgl_mask])[0])
                
        #if 'psd' in self._event_types:
            for v, d in enumerate(self._sgl_dets_to_use):

                if np.sum(~self._sgl_mask)>0:

                    loglike += self._likelihood.PG_stat(
                        self._active_time_counts_energy_psd_dict[d][~self._sgl_mask],
                        self._bkg_active_psd[d]['counts_active'][~self._sgl_mask],
                        self._bkg_active_psd[d]['error_active'][~self._sgl_mask],
                        self._expected_model_counts_psd[v][~self._sgl_mask])
                        
                    #loglike += np.sum(poisson_observed_gaussian_background(
                    #    self._active_time_counts_energy_psd_dict[d][~self._sgl_mask],
                    #    self._bkg_active_psd[d]['counts_active'][~self._sgl_mask],
                    #    self._bkg_active_psd[d]['error_active'][~self._sgl_mask],
                    #    self._expected_model_counts_psd[v][~self._sgl_mask])[0])

        if 'double' in self._event_types:
            for v, d in enumerate(self._me2_dets_to_use):

                loglike += self._likelihood.PG_stat(
                    self._active_time_counts_energy_me2_dict[d],
                    self._bkg_active_me2[d]['counts_active'],
                    self._bkg_active_me2[d]['error_active'],
                    self._expected_model_counts_me2[v])

                
                #loglike += np.sum(poisson_observed_gaussian_background(
                #    self._active_time_counts_energy_me2_dict[d],
                #    self._bkg_active_me2[d]['counts_active'],
                #    self._bkg_active_me2[d]['error_active'],
                #    self._expected_model_counts_me2[v])[0])

        if 'triple' in self._event_types:
            for v, d in enumerate(self._me3_dets_to_use):

                loglike += self._likelihood.PG_stat(
                    self._active_time_counts_energy_me3_dict[d],
                    self._bkg_active_me3[d]['counts_active'],
                    self._bkg_active_me3[d]['error_active'],
                    self._expected_model_counts_me3[v])
                    

                #loglike += np.sum(poisson_observed_gaussian_background(
                #    self._active_time_counts_energy_me3_dict[d],
                #    self._bkg_active_me3[d]['counts_active'],
                #    self._bkg_active_me3[d]['error_active'],
                #    self._expected_model_counts_me3[v])[0])

        return loglike

    def plot_ebin_lightcurve_one_det(self, det):
        """
        Plot data and bkg fits of sgl dets
        :return: fig
        """
        n_ebin = len(self._ebounds)-1
        
        fig, axes = plt.subplots(n_ebin, 1, sharex=True, figsize=(8.27, 11.69))

        if det in range(19):
            assert "single" in self._event_types, "Single detectors are not used in this analysis."
            assert det in self._sgl_dets_to_use, "This detector is not used in the analysis"
            plot_type = "single"
        elif det in range(19,61):
            assert "double" in self._event_types, "Double detectors are not used in this analysis"
            assert det in self._me2_dets_to_use, "This detector is not used in the analysis"
            plot_type = "double"
        elif det in range(61,85):
            assert "triple" in self._event_types, "Triple detectors are not used in this analysis"
            assert det in self._me3_dets_to_use, "This detector is not used in the analysis"
            plot_type = "triple"
        else:
            raise AssertionError("Please use a valid detector id between 0 and 84")

        for i in range(n_ebin):
            if plot_type=="single":
                # Plot data
                axes[i].step(self._data_object.time_bins, (self._data_object.energy_and_time_bin_sgl_dict[det][:,i]+self._data_object.energy_and_time_bin_psd_dict[det][:,i])/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='black', label='{}-{} keV'.format(int(self._ebounds[i]), int(self._ebounds[i+1])), linewidth=0.5)

                # Plot the bkg fit
                axes[i].step(self._data_object.time_bins, self._bkg_sgl_psd_sum[det]['counts'][i]/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='red')

            elif plot_type=="double":
                # Plot data
                axes[i].step(self._data_object.time_bins, (self._data_object.energy_and_time_bin_me2_dict[det][:,i])/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='black', label='{}-{} keV'.format(int(self._ebounds[i]), int(self._ebounds[i+1])), linewidth=0.5)

                # Plot the bkg fit
                axes[i].step(self._data_object.time_bins, self._bkg_me2[det]['counts'][i]/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='red')

            else:
                # Plot data
                axes[i].step(self._data_object.time_bins, (self._data_object.energy_and_time_bin_me3_dict[det][:,i])/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='black', label='{}-{} keV'.format(int(self._ebounds[i]), int(self._ebounds[i+1])), linewidth=0.5)

                # Plot the bkg fit
                axes[i].step(self._data_object.time_bins, self._bkg_me3[det]['counts'][i]/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='red')

            # Plot active time
            axes[i].axvspan(self._active_start, self._active_stop, color='green', alpha=0.2)

            # Plot bkg times
            axes[i].axvspan(self._bkg1_start, self._bkg1_stop, color='red', alpha=0.2)
            axes[i].axvspan(self._bkg2_start, self._bkg2_stop, color='red', alpha=0.2)

            axes[i].locator_params(axis='y', nbins=1)
            axes[i].legend(loc='upper right')
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_xlim(self._bkg1_start, self._bkg2_stop)

        ax_frame = fig.add_subplot(111, frameon=False)
        ax_frame.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax_frame.set_ylabel('Count rates [cnts s$^{-1}$]')
        ax_frame.set_title('Detector {}'.format(det))

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        return fig

    

    def plot_sgl_lightcurve(self):
        """
        Plot data and bkg fits of sgl dets
        :return: fig
        """

        n_dets = len(self._sgl_dets_to_use)

        n_rows = np.ceil(n_dets/2.)
        n_colums = 2
        
        fig, axes = plt.subplots(n_dets, 1, sharex=True, figsize=(8.27, 11.69))
        
        for i in range(n_dets):
            # Plot data
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), (np.sum(self._data_object.energy_and_time_bin_sgl_dict[self._sgl_dets_to_use[i]], axis=1)+np.sum(self._data_object.energy_and_time_bin_psd_dict[self._sgl_dets_to_use[i]], axis=1))/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#4a4e4d', label='Det {}'.format(self._sgl_dets_to_use[i]), zorder=3)

            # Plot the bkg fit
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._bkg_sgl_psd_sum[self._sgl_dets_to_use[i]]['counts'], axis=0)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#851e3e', zorder=4)

            # Plot active time
            axes[i].axvspan(self._active_start, self._active_stop, color='#88d8b0', zorder=2)

            # Plot bkg times
            axes[i].axvspan(self._bkg1_start, self._bkg1_stop, color='#2e003e', zorder=1)
            axes[i].axvspan(self._bkg2_start, self._bkg2_stop, color='#2e003e', zorder=1)

            axes[i].locator_params(axis='y', nbins=1)
            axes[i].legend(loc='upper right')
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_xlim(self._bkg1_start, self._bkg2_stop)

        ax_frame = fig.add_subplot(111, frameon=False)
        ax_frame.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax_frame.set_ylabel('Count rates [cnts s$^{-1}$]')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        return fig

    def plot_psd_lightcurve(self):
        """
        Plot data and bkg fits of sgl dets
        :return: fig
        """

        n_dets = len(self._sgl_dets_to_use)

        n_rows = np.ceil(n_dets/2.)
        n_colums = 2
        
        fig, axes = plt.subplots(n_dets, 1, sharex=True, figsize=(8.27, 11.69))
        
        for i in range(n_dets):
            # Plot data
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._data_object.energy_and_time_bin_psd_dict[self._psd_dets_to_use[i]], axis=1)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#4a4e4d', label='Det {}'.format(self._psd_dets_to_use[i]), zorder=3)

            # Plot the bkg fit
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._bkg_psd[self._psd_dets_to_use[i]]['counts'], axis=0)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#851e3e', zorder=4)

            # Plot active time
            axes[i].axvspan(self._active_start, self._active_stop, color='#88d8b0', zorder=2)

            # Plot bkg times
            axes[i].axvspan(self._bkg1_start, self._bkg1_stop, color='#2e003e', zorder=1)
            axes[i].axvspan(self._bkg2_start, self._bkg2_stop, color='#2e003e', zorder=1)

            axes[i].locator_params(axis='y', nbins=1)
            axes[i].legend(loc='upper right')
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_xlim(self._bkg1_start, self._bkg2_stop)

        ax_frame = fig.add_subplot(111, frameon=False)
        ax_frame.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax_frame.set_ylabel('Count rates [cnts s$^{-1}$]')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        return fig

    def plot_me2_lightcurve(self):
        """
        Plot data and bkg fits of sgl dets
        :return: fig
        """

        n_dets = len(self._me2_dets_to_use)

        n_rows = np.ceil(n_dets/2.)
        n_colums = 2
        
        fig, axes = plt.subplots(n_dets, 1, sharex=True, figsize=(8.27, 11.69))
        
        for i in range(n_dets):
            # Plot data
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._data_object.energy_and_time_bin_me2_dict[self._me2_dets_to_use[i]], axis=1)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#4a4e4d', label='Det {}'.format(self._me2_dets_to_use[i]), zorder=3)

            # Plot the bkg fit
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._bkg_me2[self._me2_dets_to_use[i]]['counts'], axis=0)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#851e3e', zorder=4)

            # Plot active time
            axes[i].axvspan(self._active_start, self._active_stop, color='#88d8b0', zorder=2)

            # Plot bkg times
            axes[i].axvspan(self._bkg1_start, self._bkg1_stop, color='#2e003e', zorder=1)
            axes[i].axvspan(self._bkg2_start, self._bkg2_stop, color='#2e003e', zorder=1)

            axes[i].locator_params(axis='y', nbins=1)
            axes[i].legend(loc='upper right') 
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_xlim(self._bkg1_start, self._bkg2_stop)

        ax_frame = fig.add_subplot(111, frameon=False)
        ax_frame.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax_frame.set_ylabel('Count rates [cnts s$^{-1}$]')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        return fig

    def plot_me3_lightcurve(self):
        """
        Plot data and bkg fits of sgl dets
        :return: fig
        """

        n_dets = len(self._me3_dets_to_use)

        n_rows = np.ceil(n_dets/2.)
        n_colums = 2
        
        fig, axes = plt.subplots(n_dets, 1, sharex=True, figsize=(8.27, 11.69))
        
        for i in range(n_dets):
            # Plot data
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._data_object.energy_and_time_bin_me3_dict[self._me3_dets_to_use[i]], axis=1)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#4a4e4d', label='Det {}'.format(self._me3_dets_to_use[i]), zorder=3)

            # Plot the bkg fit
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._bkg_me3[self._me3_dets_to_use[i]]['counts'], axis=0)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='#851e3e', zorder=4)

            # Plot active time
            axes[i].axvspan(self._active_start, self._active_stop, color='#88d8b0', zorder=2)

            # Plot bkg times
            axes[i].axvspan(self._bkg1_start, self._bkg1_stop, color='#2e003e', zorder=1)
            axes[i].axvspan(self._bkg2_start, self._bkg2_stop, color='#2e003e', zorder=1)

            axes[i].locator_params(axis='y', nbins=1)
            axes[i].legend(loc='upper right')
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_xlim(self._bkg1_start, self._bkg2_stop)
        
        ax_frame = fig.add_subplot(111, frameon=False)
        ax_frame.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax_frame.set_ylabel('Count rates [cnts s$^{-1}$]')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        return fig
    
    def save_fit_results_files(self, post_equal_weights_file_path, likelihood_model, overwrite=False):
        """
        Saves all the needed files to reconstruct the result. These files are necessary to create the plots with spi_display.py
        :return:
        """
        self._data_save_folder = os.path.join(get_path_of_external_data_dir(), self._analysis,
                                              self._analysis_name)
        self._data_save_folder_basis = os.path.join(get_path_of_external_data_dir(), self._analysis)

        if not os.path.exists(self._data_save_folder_basis):
            os.mkdir(self._data_save_folder_basis)

        
        if not os.path.exists(self._data_save_folder):
            os.mkdir(self._data_save_folder)
        elif not overwrite:
            raise AssertionError('The folder in which you want to save the data already exist. I do not want to overwrite it.')
        else:
            shutil.rmtree(self._data_save_folder)
            os.mkdir(self._data_save_folder)

        

            
        total_active_time = np.sum(self._active_time_bins[:,1]-self._active_time_bins[:,0])

        # PPC fit count spectrum

        sample_parameters = self._loadtxt2d(post_equal_weights_file)[:,:-1]

        # get counts for all sample parameters and the likelihood_model
        n_ppc = 100

        mask = np.zeros(len(sample_parameters), dtype=int)
        mask[:n_ppc] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)

        masked_parameter_samples = sample_parameters[mask]

        # mask the sample parameter values
        model_counts = np.empty((n_ppc, len(self._sgl_dets_to_use), len(self._ebounds)-1))
        for i in range(n_ppc):
            likelihood_model.set_free_parameters(masked_parameter_samples[i])
            self._get_model_counts(likelihood_model)
            model_counts[i] = self._expected_model_counts

        # poisson noise
        model_counts = np.random.poisson(model_counts)

        # BKG wit gaussian noise
        bkg_counts = np.empty((n_ppc, len(self._sgl_dets_to_use), len(self._ebounds)-1))

        for j, d in enumerate(self._sgl_dets_to_use):
            bkg_counts[:,j,:] = np.random.normal(self._bkg_active[d]['counts_active'], self._bkg_active[d]['error_active'], size=(n_ppc,len(self._ebounds)-1))
        # Save post equal weights file
        copyfile(post_equal_weights_file_path, os.path.join(self._data_save_folder, 'post_equal_weights.dat'))

        # Save likelihood_model as yaml

        likelihood_model.save(os.path.join(self._data_save_folder, 'likelihood_model.yaml'))

        # Save bkg counts and errors as well as the data counts for all dets and all echans
        # Also the total exposure time and the ebounds in hdf5 file
        
        with h5py.File(os.path.join(self._data_save_folder, 'data_and_background_info.h5'), 'w') as f:
            f.attrs['Analysis Type'] = self._analysis
            f.create_dataset('Ebounds', data=self._ebounds)

            if 'single' in self._event_types:
                sgl_dets = self._sgl_dets_to_use
            else:
                sgl_dets = 0

            f.create_dataset('Used Single Dets', data=sgl_dets)

            #data = f.create_group('Data')
            #bkg = f.create_group('Background')


            if 'single' in self._event_types:
                index=0
                for j in range(19):

                    # Check if det was used in the fit
                    if j in self._sgl_dets_to_use:
                        data_counts_det = self._active_time_counts_energy_sgl_dict[j]
                        model_counts_det = model_counts[:,index,:]
                        bkg_counts_det = bkg_counts[:,index,:]

                        index+=1

                    else:
                        data_counts_det = np.zeros_like(self._active_time_counts_energy_sgl_dict[self._active_time_counts_energy_sgl_dict.keys()[0]])
                        model_counts_det = np.zeros_like(model_counts[:,0,:])
                        bkg_counts_det = np.zeros_like(bkg_counts[:,0,:])


                    grp = f.create_group('Detector {}'.format(j))
                    grp.create_dataset('Detected counts', data=active_data)
                    grp.create_dataset('Model counts ppc', data=model_counts_det)
                    grp.create_dataset('Background counts ppc', data=bkg_counts_det)

                """
                for j in range(19):
                    det = f.create_group('Detector {}'.format(j))
                    if j in self._sgl_dets:
                        data_counts_det = self._active_time_counts_energy_sgl_dict[j]
                        bkg_counts_det = self._bkg_active[j]['counts_active']
                        bkg_error_det = self._bkg_active[j]['error_active']
                    else:
                        data_counts_det = 0
                        bkg_counts_det = 0
                        bkg_error_det = 0

                    det.create_dataset('Detected Counts', data=data_counts_det)
                    det.create_dataset('Background Counts', data=bkg_counts_det)
                    det.create_dataset('Background Counts Error', data=bkg_error_det)
                 """

    @property
    def data_save_folder(self):
        return self._data_save_folder
        
    def _loadtxt2d(self, intext):
        try:
            return np.loadtxt(intext, ndmin=2)
        except:
            return np.loadtxt(intext)

    
    def plot_fit_data_sgl(self, post_equal_weights_file, likelihood_model):
        """
        Plot the data and the fit result in one plot
        :param post_equal_weights_file: file with post_equal_weights parameter samples - for example from multinest
        :return: fig
        """

        total_active_time = np.sum(self._active_time_bins[:,1]-self._active_time_bins[:,0])

        # PPC fit count spectrum

        sample_parameters = self._loadtxt2d(post_equal_weights_file)[:,:-1]

        # Check if there is a psd_eff parametert in the post_equal_weights file
        if sample_parameters.shape[1]>len(likelihood_model.free_parameters):
            # psd_eff_fit_array 
            psd_eff_fit_array = sample_parameters[:,-1]
            sample_parameters = sample_parameters[:,:-1]
            psd_eff_variable = True
        else:
            psd_eff_variable = False
            
        # get counts for all sample parameters and the likelihood_model
        n_ppc = 100

        mask = np.zeros(len(sample_parameters), dtype=int)
        mask[:n_ppc] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)

        masked_parameter_samples = sample_parameters[mask]
        if psd_eff_variable:
            masked_psd_eff_fit_array = psd_eff_fit_array[mask]
        # mask the sample parameter values
        model_counts = np.empty((n_ppc, len(self._sgl_dets_to_use), len(self._ebounds)-1))
        for i in range(n_ppc):
            likelihood_model.set_free_parameters(masked_parameter_samples[i])
            if psd_eff_variable:
                self.set_psd_eff(masked_psd_eff_fit_array[i])
            self._get_model_counts(likelihood_model)
            model_counts[i] = self._expected_model_counts_sgl_psd_sum

        # poisson noise
        #model_counts = np.random.poisson(model_counts)

        # BKG wit gaussian noise
        bkg_counts = np.empty((n_ppc, len(self._sgl_dets_to_use), len(self._ebounds)-1))

        for j, d in enumerate(self._sgl_dets_to_use):
            bkg_counts[:,j,:] = np.random.normal(self._bkg_active_sgl_psd_sum[d]['counts_active'], self._bkg_active_sgl_psd_sum[d]['error_active'], size=(n_ppc,len(self._ebounds)-1))

        # Set negative bkg to zeros
        bkg_counts = np.where(bkg_counts<0, 0, bkg_counts)

        # Ebin sizes
        ebin_sizes = self._ebounds[1:]-self._ebounds[:-1]
        
        # Fitted GRB count spectrum ppc versus count space data of all dets
        if 'single' in self._event_types:

            # Init figure
            nrows = 4
            ncol = 5
            fig, axes = plt.subplots(nrows, ncol, sharex=True, sharey=True, figsize=(8.27, 11.69))
            axes_array = axes.flatten()
            index = 0
            red_indices = []
            # Loop over all possible single dets
            for j in range(19):
                plot_number = j#(j*4)/19
                if (plot_number/float(ncol)).is_integer():
                        axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

                # If this single det was used plot the fit vs. data
                if j in self._sgl_dets_to_use:

                    # Data rate
                    active_data = self._active_time_counts_energy_sgl_psd_sum_dict[j]/total_active_time#or axis=1

                    # PPC fit count spectrum
                    # get counts for all sample parameters and the likelihood_model
                    # Add poisson noise
                    model_bkg_rates_det = np.random.poisson((model_counts[:,index,:]+bkg_counts[:,index,:])/total_active_time) 

                    q_levels = [0.68,0.95, 0.99]
                    colors = ['#354458', '#3A9AD9', '#29ABA4']#['#588C73', '#85C4B9', '#8C4646']# TODO change this to more fancy colors

                    # get 68 and 95 % boundaries and plot them
                    for i,level in enumerate(q_levels):
                        low = np.percentile(model_bkg_rates_det/ebin_sizes, 50-50*level, axis=0)
                        high = np.percentile(model_bkg_rates_det/ebin_sizes, 50+50*level, axis=0)
                        axes_array[plot_number].fill_between(self._ebounds[1:],
                                                       low,
                                                       high,
                                                       alpha=0.5,
                                                       color=colors[i],
                                                       zorder=10-i,
                                                       step='post')

                    axes_array[plot_number].step(self._ebounds[1:],
                                           active_data/ebin_sizes,
                                           where='post',
                                           color='black',
                                                 zorder=19,
                                           label='Detector {}'.format(j))
                    if (plot_number/float(ncol)).is_integer():
                        axes_array[plot_number].set_ylabel('Count rate [cts $s^{-1}$ $keV^{-1}$]')

                    index+=1
                # If det not used only plot legend entry with remark "not used or defect"
                else:
                    red_indices.append(j)
                    axes_array[plot_number].plot([], [], ' ', label='Detector {} \n Not used'.format(j))

            # Make legend and mark the not used dets red
            for i, ax in enumerate(axes.flatten()):
                ax.set_xlabel('Energy [keV]')
                l = ax.legend()
                if i in red_indices:
                    l.get_texts()[0].set_color("red")
                ax.set_xscale('log')
                #ax.set_yscale('log')
            fig.tight_layout()
            fig.subplots_adjust(hspace=0, wspace=0) 
            fig.savefig('data_plot.pdf')

    def plot_fit_data_psd(self, post_equal_weights_file, likelihood_model):
        """
        Plot the data and the fit result in one plot
        :param post_equal_weights_file: file with post_equal_weights parameter samples - for example from multinest
        :return: fig
        """

        total_active_time = np.sum(self._active_time_bins[:,1]-self._active_time_bins[:,0])

        # PPC fit count spectrum

        sample_parameters = self._loadtxt2d(post_equal_weights_file)[:,:-1]
        
        # Check if there is a psd_eff parametert in the post_equal_weights file
        if sample_parameters.shape[1]>len(likelihood_model.free_parameters):
            # psd_eff_fit_array 
            psd_eff_fit_array = sample_parameters[:,-1]
            sample_parameters = sample_parameters[:,:-1]
            psd_eff_variable = True
        else:
            psd_eff_variable = False

        # get counts for all sample parameters and the likelihood_model
        n_ppc = 100

        mask = np.zeros(len(sample_parameters), dtype=int)
        mask[:n_ppc] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)

        masked_parameter_samples = sample_parameters[mask]

        # mask the sample parameter values
        model_counts = np.empty((n_ppc, len(self._sgl_dets_to_use), len(self._ebounds)-1))
        for i in range(n_ppc):
            likelihood_model.set_free_parameters(masked_parameter_samples[i])
            self._get_model_counts(likelihood_model)
            model_counts[i] = self._expected_model_counts_psd

        # poisson noise
        #model_counts = np.random.poisson(model_counts)

        # BKG wit gaussian noise
        bkg_counts = np.empty((n_ppc, len(self._sgl_dets_to_use), len(self._ebounds)-1))

        for j, d in enumerate(self._psd_dets_to_use):
            bkg_counts[:,j,:] = np.random.normal(self._bkg_active_psd[d]['counts_active'], self._bkg_active_psd[d]['error_active'], size=(n_ppc,len(self._ebounds)-1))

        # Set negative bkg to zeros
        bkg_counts = np.where(bkg_counts<0, 0, bkg_counts)

        # Fitted GRB count spectrum ppc versus count space data of all dets

        # Init figure
        nrows = 4
        ncol = 5
        fig, axes = plt.subplots(nrows, ncol, sharex=True, sharey=True, figsize=(8.27, 11.69))
        axes_array = axes.flatten()
        index = 0
        red_indices = []
        # Loop over all possible single dets
        for j in range(19):
            plot_number = j#(j*4)/19
            if (plot_number/float(ncol)).is_integer():
                    axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

            # If this single det was used plot the fit vs. data
            if j in self._psd_dets_to_use:

                # Data rate
                active_data = self._active_time_counts_energy_psd_dict[j]/total_active_time#or axis=1

                # PPC fit count spectrum
                # get counts for all sample parameters and the likelihood_model
                model_bkg_rates_det = np.random.poisson(model_counts[:,index,:]+bkg_counts[:,index,:])/total_active_time 

                q_levels = [0.68,0.95, 0.99]
                colors = ['#354458', '#3A9AD9', '#29ABA4']

                # get 68 and 95 % boundaries and plot them
                for i,level in enumerate(q_levels):
                    low = np.percentile(model_bkg_rates_det, 50-50*level, axis=0)
                    high = np.percentile(model_bkg_rates_det, 50+50*level, axis=0)
                    axes_array[plot_number].fill_between(self._ebounds[1:],
                                                   low,
                                                   high,
                                                   color=colors[i],
                                                   alpha=0.5,
                                                   zorder=10-i,
                                                   step='post')

                axes_array[plot_number].step(self._ebounds[1:],
                                       active_data,
                                       where='post',
                                       color='black',
                                             zorder=19,
                                       label='Detector {}'.format(j))
                if (plot_number/float(ncol)).is_integer():
                    axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

                index+=1
            # If det not used only plot legend entry with remark "not used or defect"
            else:
                red_indices.append(j)
                axes_array[plot_number].plot([], [], ' ', label='Detector {} \n Not used'.format(j))

        # Make legend and mark the not used dets red
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlabel('Energy [keV]')
            l = ax.legend()
            if i in red_indices:
                l.get_texts()[0].set_color("red")
            ax.set_xscale('log')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0) 
        fig.savefig('data_plot.pdf')

    def plot_fit_data_me2(self, post_equal_weights_file, likelihood_model):
        """
        Plot the data and the fit result in one plot
        :param post_equal_weights_file: file with post_equal_weights parameter samples - for example from multinest
        :return: fig
        """

        total_active_time = np.sum(self._active_time_bins[:,1]-self._active_time_bins[:,0])

        # PPC fit count spectrum

        sample_parameters = self._loadtxt2d(post_equal_weights_file)[:,:-1]

        # Check if there is a psd_eff parametert in the post_equal_weights file
        if sample_parameters.shape[1]>len(likelihood_model.free_parameters):
            # psd_eff_fit_array 
            psd_eff_fit_array = sample_parameters[:,-1]
            sample_parameters = sample_parameters[:,:-1]
            psd_eff_variable = True
        else:
            psd_eff_variable = False

        # get counts for all sample parameters and the likelihood_model
        n_ppc = 100

        mask = np.zeros(len(sample_parameters), dtype=int)
        mask[:n_ppc] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)

        masked_parameter_samples = sample_parameters[mask]

        # mask the sample parameter values
        model_counts = np.empty((n_ppc, len(self._me2_dets_to_use), len(self._ebounds)-1))
        for i in range(n_ppc):
            likelihood_model.set_free_parameters(masked_parameter_samples[i])
            self._get_model_counts(likelihood_model)
            model_counts[i] = self._expected_model_counts_me2

        # poisson noise
        #model_counts = np.random.poisson(model_counts)

        # BKG wit gaussian noise
        bkg_counts = np.empty((n_ppc, len(self._me2_dets_to_use), len(self._ebounds)-1))

        for j, d in enumerate(self._me2_dets_to_use):
            bkg_counts[:,j,:] = np.random.normal(self._bkg_active_me2[d]['counts_active'], self._bkg_active_me2[d]['error_active'], size=(n_ppc,len(self._ebounds)-1))

        # Set negative bkg to zeros
        bkg_counts = np.where(bkg_counts<0, 0, bkg_counts)
            
        # Init figure
        nrows = 6
        ncol = 7
        fig, axes = plt.subplots(nrows, ncol, sharex=True, sharey=True, figsize=(8.27, 11.69))
        axes_array = axes.flatten()
        index = 0
        red_indices = []
        # Loop over all possible single dets
        for j in range(19,61):
            plot_number = j-19#(j*4)/19
            if (plot_number/float(ncol)).is_integer():
                    axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

            # If this single det was used plot the fit vs. data
            if j in self._me2_dets_to_use:

                # Data rate
                active_data = self._active_time_counts_energy_me2_dict[j]/total_active_time#or axis=1

                # PPC fit count spectrum
                # get counts for all sample parameters and the likelihood_model
                model_bkg_rates_det = np.random.poisson(model_counts[:,index,:]+bkg_counts[:,index,:])/total_active_time 

                q_levels = [0.68,0.95, 0.99]
                colors = ['#354458', '#3A9AD9', '#29ABA4']

                # get 68 and 95 % boundaries and plot them
                for i,level in enumerate(q_levels):
                    low = np.percentile(model_bkg_rates_det, 50-50*level, axis=0)
                    high = np.percentile(model_bkg_rates_det, 50+50*level, axis=0)
                    axes_array[plot_number].fill_between(self._ebounds[1:],
                                                   low,
                                                   high,
                                                   color=colors[i],
                                                   alpha=0.5,
                                                         zorder=10-i,
                                                   step='post')

                axes_array[plot_number].step(self._ebounds[1:],
                                       active_data,
                                       where='post',
                                       color='black',
                                             zorder=19,
                                       label='Detector {}'.format(j))
                if (plot_number/float(ncol)).is_integer():
                    axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

                index+=1
            # If det not used only plot legend entry with remark "not used or defect"
            else:
                red_indices.append(j-19)
                axes_array[plot_number].plot([], [], ' ', label='Detector {} \n Not used'.format(j))

        # Make legend and mark the not used dets red
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlabel('Energy [keV]')
            l = ax.legend()
            if i in red_indices:
                l.get_texts()[0].set_color("red")
            ax.set_xscale('log')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0) 
        fig.savefig('data_plot.pdf')

    def plot_fit_data_me3(self, post_equal_weights_file, likelihood_model):
        """
        Plot the data and the fit result in one plot
        :param post_equal_weights_file: file with post_equal_weights parameter samples - for example from multinest
        :return: fig
        """

        total_active_time = np.sum(self._active_time_bins[:,1]-self._active_time_bins[:,0])

        # PPC fit count spectrum

        sample_parameters = self._loadtxt2d(post_equal_weights_file)[:,:-1]

        # Check if there is a psd_eff parametert in the post_equal_weights file
        if sample_parameters.shape[1]>len(likelihood_model.free_parameters):
            # psd_eff_fit_array 
            psd_eff_fit_array = sample_parameters[:,-1]
            sample_parameters = sample_parameters[:,:-1]
            psd_eff_variable = True
        else:
            psd_eff_variable = False

        # get counts for all sample parameters and the likelihood_model
        n_ppc = 100

        mask = np.zeros(len(sample_parameters), dtype=int)
        mask[:n_ppc] = 1
        np.random.shuffle(mask)
        mask = mask.astype(bool)

        masked_parameter_samples = sample_parameters[mask]

        # mask the sample parameter values
        model_counts = np.empty((n_ppc, len(self._me3_dets_to_use), len(self._ebounds)-1))
        for i in range(n_ppc):
            likelihood_model.set_free_parameters(masked_parameter_samples[i])
            self._get_model_counts(likelihood_model)
            model_counts[i] = self._expected_model_counts_me3

        # poisson noise
        #model_counts = np.random.poisson(model_counts)

        # BKG wit gaussian noise
        bkg_counts = np.empty((n_ppc, len(self._me3_dets_to_use), len(self._ebounds)-1))

        for j, d in enumerate(self._me3_dets_to_use):
            bkg_counts[:,j,:] = np.random.normal(self._bkg_active_me3[d]['counts_active'], self._bkg_active_me3[d]['error_active'], size=(n_ppc,len(self._ebounds)-1))
            
        # Set negative bkg to zeros
        bkg_counts = np.where(bkg_counts<0, 0, bkg_counts)
            
        # Init figure
        nrows = 4
        ncol = 6
        fig, axes = plt.subplots(nrows, ncol, sharex=True, sharey=True, figsize=(8.27, 11.69))
        axes_array = axes.flatten()
        index = 0
        red_indices = []
        # Loop over all possible single dets
        for j in range(61,85):
            plot_number = j-61#(j*4)/19
            if (plot_number/float(ncol)).is_integer():
                    axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

            # If this single det was used plot the fit vs. data
            if j in self._me3_dets_to_use:

                # Data rate
                active_data = self._active_time_counts_energy_me3_dict[j]/total_active_time#or axis=1

                # PPC fit count spectrum
                # get counts for all sample parameters and the likelihood_model
                model_bkg_rates_det = np.random.poisson(model_counts[:,index,:]+bkg_counts[:,index,:])/total_active_time 

                q_levels = [0.68,0.95, 0.99]
                colors = ['#354458', '#3A9AD9', '#29ABA4']

                # get 68 and 95 % boundaries and plot them
                for i,level in enumerate(q_levels):
                    low = np.percentile(model_bkg_rates_det, 50-50*level, axis=0)
                    high = np.percentile(model_bkg_rates_det, 50+50*level, axis=0)
                    axes_array[plot_number].fill_between(self._ebounds[1:],
                                                   low,
                                                   high,
                                                   color=colors[i],
                                                   alpha=0.5,
                                                   zorder=10-i,
                                                   step='post')

                axes_array[plot_number].step(self._ebounds[1:],
                                       active_data,
                                       where='post',
                                       color='black',
                                             zorder=19,
                                       label='Detector {}'.format(j))
                if (plot_number/float(ncol)).is_integer():
                    axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')

                index+=1
            # If det not used only plot legend entry with remark "not used or defect"
            else:
                red_indices.append(j-61)
                axes_array[plot_number].plot([], [], ' ', label='Detector {} \n Not used'.format(j))

        # Make legend and mark the not used dets red
        for i, ax in enumerate(axes.flatten()):
            ax.set_xlabel('Energy [keV]')
            l = ax.legend()
            if i in red_indices:
                l.get_texts()[0].set_color("red")
            ax.set_xscale('log')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0, wspace=0) 
        fig.savefig('data_plot.pdf')

class GRBAnalysisRMF(GRBAnalysis):

    def __init__(self, configuration, likelihood_model):
        """
        Init GRB analysis if the full RMF shoud be used in the fit (slower but correct).

        :param configuration: Configuration setup as dict
        :param likelihood_model: Astromodel instance describing the used model
        :return:
        """
        
        super(GRBAnalysisRMF, self).__init__(configuration, likelihood_model)

    def _fold(self, response, spectrum_bins):
        """
        Get the counts in all Ebins for given response (matrix in this case) and 
        integrated flux in the defined spectrum bins.
        
        :param response: Response Matrix
        :param spectrum_bins: Integrated flux in defined spectral bins
        """
        return np.dot(response, spectrum_bins)

    def _init_response(self):
        """
        Initalize the response object with RMF
        :return:
        """

        self._response_object = ResponseRMF(ebounds=self._ebounds, time=self._time_of_grb)

    
class GRBAnalysisPhotopeak(GRBAnalysis):

    def __init__(self, configuration, likelihood_model):
        """
        Init GRB analysis if only the photopeak eff area shoud be used in the 
        fit (faster but not really  correct).

        :param configuration: Configuration setup as dict
        :param likelihood_model: Astromodel instance describing the used model
        :return:
        """
        
        super(GRBAnalysisPhotopeak, self).__init__(configuration, likelihood_model)

    def _fold(self, response, spectrum_bins):
        """
        Get the counts in all Ebins for given response (array in this case) and 
        integrated flux in the defined spectrum bins.
        
        :param response: Photopeak Response Array
        :param spectrum_bins: Integrated flux in defined spectral bins
        """
        return np.multiply(response, spectrum_bins)

    def _init_response(self):
        """
        Initalize the response object without RMF
        :return:
        """

        self._response_object = ResponsePhotopeak(ebounds=self._ebounds, time=self._time_of_grb)
