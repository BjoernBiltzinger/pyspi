from astromodels import *
import astropy.units as u
from astropy.coordinates import ICRS, Galactic, SkyCoord
from astropy.time.core import Time
from datetime import datetime
import os
from shutil import copyfile, rmtree
import sys

from pyspi.io.package_data import get_path_of_external_data_dir

from pyspi.spi_data import *
from pyspi.spi_response import ResponsePhotopeak, ResponseRMF
from pyspi.spi_pointing import *
from pyspi.spi_frame import *
from pyspi.utils.likelihood import Likelihood
from pyspi.data_constant_sources import DataConstantSources
from pyspi.background_model import BackgroundModel

class SPIConstantSourceAnalysis(object):

    def __init__(self, configuration, likelihood_model):
        """
        Init a Spi Analysis object for an analysis of a constant source 
        (Point sources and extended sources). Superclass of SPIAnalysis.
        :param configuration: Configuration dictionary
        :param likelihood_model: The inital astromodels likelihood_model
        """
        #raise NotImplementedError('Constant source analysis not implemented at the moment!') 
        # TODO: Add a test if the configuration file is valid for a GRB analysis
        # Which event types should be used?
        self._event_types = configuration['Event_types']

        # Which energy range?
        self._emin = float(configuration['emin'])
        self._emax = float(configuration['emax'])
        self._pointings_list = np.array(configuration['Pointings'])#, '010200040010']) #DUMMY VALUE
        self._active_time_seconds = 1000*np.ones((self._pointings_list.size, 19)) #DUMMMY VALUE
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
        # Set bkg norm
        self._beta = 0.99
        # Simmulate a GRB at the given time? Only used for testing!
        self._simulate = configuration['Simulate']

        if self._simulate is not None:
            raise NotImplementedError('Simulate Signal not implemented for constant sources yet.')

                # Get the unique name for the analysis (this name will be used to save the results later)
        self._analysis_name = configuration['Unique_analysis_name']

        # Check if the unique name is given and if it was never used before.
        # If not add a time stamp to it to avoid
        # overwritting old results
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
        #self.get_background_estimation()

        # Init the SPI frame (to transform J2000 to SPI coordinates)
        self._init_frames()

        # Set the model 
        self.set_model(likelihood_model)

        # Set Likelihood class
        self._init_likelihood()


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
            for i, e in enumerate(self._ebounds):
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
            for i, e in enumerate(self._ebounds):
                if e>=psd_high_energy and not stop_found:
                    stop_index = i
                    stop_found=True
            self._ebounds = np.insert(self._ebounds, stop_index, psd_high_energy)
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
                #bad_sgl_dets = self._data_object.bad_sgl_dets
                
                self._sgl_dets_to_use = np.arange(0,19,1,dtype=int)#[~bad_sgl_dets]

            # If all dets should be used we will just ignore the ones that are turned off
            #if "psd" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                #bad_psd_dets = self._data_object.bad_psd_dets
                
                #self._psd_dets_to_use = np.arange(0,19,1,dtype=int)[~bad_sgl_dets]
                
            if "double" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                #bad_me2_dets = self._data_object.bad_me2_dets
                
                self._me2_dets_to_use = np.arange(19,61,1,dtype=int)#[~bad_me2_dets]
                
            if "triple" in self._event_types:
                # Get a mask of the dets that are turned off. (Dets with 0 counts)
                #bad_me3_dets = self._data_object.bad_me3_dets

                self._me3_dets_to_use = np.arange(61,85,1,dtype=int)#[~bad_me3_dets]
            
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


    def _init_likelihood(self):
        """
        Initalize Likelihood object. Will be used later to calculate the sum of all log-likelihoods
        :return:
        """
        self._likelihood = Likelihood(numba_cpu=True, parallel=True)

    def _init_data(self):
        """
        Get the data object with all the data we need
        :return:
        """
        

        self._data_object = DataConstantSources(pointings_list=self._pointings_list,
                                                event_types=self._event_types,
                                                afs=True,
                                                ebounds=self._ebounds)

        if self._binned:
            pass
        else:
            raise NotImplementedError('Only binned analysis implemented at the moment!')

    def _init_frames(self):
        """
        Initalize the spi frame object for every pointing
        :return:
        """
        self._frame_object_list = []
        self._pointing_icrs_list = []
        for path in self._data_object.geometry_file_paths:
            pointing_object = SPIPointing(path)

            self._frame_object_list.append(SPIFrame(**pointing_object.sc_points[10]))

            # get skycoord object of center ra and dec in icrs frame
            pointing_sat = SkyCoord(lon=0, lat=0, unit='deg', frame=self._frame_object_list[-1])

            self._pointing_icrs_list.append(pointing_sat.transform_to('icrs'))

    def set_psd_eff(self, value):
        """
        Set the psd efficency
        :param value: Value for psd eff
        :return:
        """
        self._eff_psd = value

    def set_bkg_norm(self, value):
        """
        Set the psd efficency
        :param value: Value for psd eff
        :return:
        """
        self._beta = value

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

        self._response_object = ResponseRMF(ebounds=self._ebounds)

            
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
        self._point_sources[name] = {}
        
        for n in range(self._pointings_list.size):
            # ra and dec to sat coord
            icrscoord = SkyCoord(ra=point_source.position.ra.value,
                                 dec=point_source.position.dec.value,
                                 unit='deg',
                                 frame='icrs')

            satcoord = icrscoord.transform_to(self._frame_object_list[n])

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
                    predicted_count_rates_sgl[d] = self._fold(response_sgl[d], spectrum_bins)*self._active_time_seconds[n, d]

            #if 'psd' in self._event_types:
                for d in self._sgl_dets_to_use:
                    predicted_count_rates_psd[d] = self._fold(response_psd[d], spectrum_bins)*self._active_time_seconds[n,d]

            if 'double' in self._event_types:
                for d in self._me2_dets_to_use:
                    predicted_count_rates_me2[d] = self._fold(response_me2[d], spectrum_bins)*self._active_time_seconds[n,d]

            if 'triple' in self._event_types:
                for d in self._me3_dets_to_use:
                    predicted_count_rates_me3[d] = self._fold(response_me3[d], spectrum_bins)*self._active_time_seconds[n,d]

            spectal_parameters = {}

            for i, component in enumerate(point_source._components.values()):

                spectral_parameters_component = {}
                for key in component.shape.parameters.keys():
                    spectral_parameters_component[key] = component.shape.parameters[key].value

                spectal_parameters[i] = spectral_parameters_component

            # Create the entry of the point source
            self._point_sources[name][n] = {'ra': point_source.position.ra.value,
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
        for n in range(self._pointings_list.size):
            if point_source.position.ra.value != self._point_sources[name][n]['ra'] or \
               point_source.position.dec.value != self._point_sources[name][n]['dec']:

                # ra and dec to sat coord
                icrscoord = SkyCoord(ra=point_source.position.ra.value,
                                     dec=point_source.position.dec.value,
                                     unit='deg',
                                     frame='icrs')
            
                satcoord = icrscoord.transform_to(self._frame_object_list[n])

                ra_sat = satcoord.lon.deg
                dec_sat = satcoord.lat.deg
                print(ra_sat, dec_sat)
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
                response_sgl = self._point_sources[name][n]['response_sgl']
                response_psd = {}
                for key in response_sgl:
                    response_psd[key] = response_sgl[key]*self._eff_psd
                response_me2 = self._point_sources[name][n]['response_me2']
                response_me3 = self._point_sources[name][n]['response_me3']

            # Get current spectrum of source
            spectrum_bins = self.calculate_spectrum(point_source)

            predicted_count_rates_sgl = {}
            predicted_count_rates_psd = {}
            predicted_count_rates_me2 = {}
            predicted_count_rates_me3 = {}
            # Get the predicted count rates in all dets in all PHA bins (individual for all pointings later)
            if 'single' in self._event_types:
                for d in self._sgl_dets_to_use:
                    predicted_count_rates_sgl[d] = self._fold(response_sgl[d], spectrum_bins)*self._active_time_seconds[n,d]
                for d in self._sgl_dets_to_use:
                    predicted_count_rates_psd[d] = self._fold(response_psd[d], spectrum_bins)*self._active_time_seconds[n,d]

            if 'double' in self._event_types:
                for d in self._me2_dets_to_use:
                    predicted_count_rates_me2[d] = self._fold(response_me2[d], spectrum_bins)*self._active_time_seconds[n,d]

            if 'triple' in self._event_types:
                for d in self._me3_dets_to_use:
                    predicted_count_rates_me3[d] = self._fold(response_me3[d], spectrum_bins)*self._active_time_seconds[n,d]

            spectal_parameters = {}
            for i, component in enumerate(point_source._components.values()):

                spectral_parameters_component = {}
                for key in component.shape.parameters.keys():
                    spectral_parameters_component[key] = component.shape.parameters[key].value

                spectal_parameters[i] = spectral_parameters_component

            # Update the entry of the point source
            self._point_sources[name][n] = {'ra': point_source.position.ra.value,
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


    def _get_model_counts(self, likelihood_model):
        """
        Get the current model counts for all dets #TODO simplify this to only one possible source(This is a GRB analysis)
        :param likelihood_model: current likelihood_model
        :return:
        """
        
        self.update_model(likelihood_model)
        for n in range(self._pointings_list.size):
            if "single" in self._event_types:
                expected_model_counts = np.zeros((self._pointings_list.size,
                                                  len(self._sgl_dets_to_use),
                                                  len(self._data_object.ene_min)))
                for i, d in enumerate(self._sgl_dets_to_use):
                    for point_s in self._point_sources.keys():
                        expected_model_counts[n,i] += self._point_sources[point_s]['predicted_count_rates_sgl'][d]

                self._expected_model_counts_sgl_psd_sum = expected_model_counts

            #if "psd" in self._event_types:
                expected_model_counts = np.zeros((self._pointings_list.size,
                                                  len(self._sgl_dets_to_use),
                                                  len(self._data_object.ene_min)))
                for i, d in enumerate(self._sgl_dets_to_use):
                    for point_s in self._point_sources.keys():
                        expected_model_counts[n, i] += self._point_sources[point_s]['predicted_count_rates_psd'][d]

                self._expected_model_counts_psd = expected_model_counts

            if "double" in self._event_types:
                expected_model_counts = np.zeros((self._pointings_list.size,
                                                  len(self._me2_dets_to_use),
                                                  len(self._data_object.ene_min)))
                for i, d in enumerate(self._me2_dets_to_use):
                    for point_s in self._point_sources.keys():
                        expected_model_counts[n, i] += self._point_sources[point_s]['predicted_count_rates_me2'][d]

                self._expected_model_counts_me2 = expected_model_counts

            if "triple" in self._event_types:
                expected_model_counts = np.zeros((self._pointings_list.size,
                                                  len(self._me3_dets_to_use),
                                                  len(self._data_object.ene_min)))
                for i, d in enumerate(self._me3_dets_to_use):
                    for point_s in self._point_sources.keys():
                        expected_model_counts[n, i] += self._point_sources[point_s]['predicted_count_rates_me3'][d]

                self._expected_model_counts_me3 = expected_model_counts

    def get_background_estimation(self):
        """
        Use the background model to get the background response for all pointings all used dets.
        During the fit we will multiply this background response with the beta(E) term, 
        that descibes how much of the total signal is due to the background
        :return:
        """
        backgroundmodel = BackgroundModel(self._pointings_list,
                                          event_types=self._event_types,
                                          ebounds=self._ebounds)

        if "single" in self._event_types:
            bkg_base_response_rate_sgl_psd = backgroundmodel._build_base_background_responses()

            livetime_rev_sgl = backgroundmodel._livetime_orbit()

            ratio_livetimes = np.divide(self._active_time_seconds, livetime_rev_sgl)

            bkg_base_response_sgl_psd_time_corr = np.multiply(bkg_base_response_rate_sgl_psd.T,
                                                          ratio_livetimes.T).T
            
            bkg_base_response_rate_sgl_psd_sum = np.sum(bkg_base_response_sgl_psd_time_corr, axis=(1,2))
            data_sglpsd_sum = np.sum(self._data_object.counts_sgl_with_psd, axis=(1,2))

            ratio_norm = data_sglpsd_sum/bkg_base_response_rate_sgl_psd_sum
            
            self._bkg_response_sgl_psd_base = np.multiply(bkg_base_response_sgl_psd_time_corr.T, ratio_norm).T
            
        if "double" in self._event_types:
            raise NotImplementedError('Background model only iplemented for singles.')
            
        
                
    def get_log_like(self, likelihood_model):
        """
        Return the current log likelihood value, only point_source at the moment
        :return: loglike value
        """
        
        self._get_model_counts(likelihood_model)

        # Use pgstat likelihood (background gaussian + signal poisson)
        loglike = 0
        if 'single' in self._event_types:
            for n in range(self._pointings_list.size):
                
                for v, d in enumerate(self._sgl_dets_to_use):

                    loglike += self._likelihood.Cash_stat(
                        self._active_time_counts_energy_sgl_psd_sum_dict[n,d][self._sgl_mask],
                        self._expected_model_counts_sgl_psd_sum[n,v][self._sgl_mask]+
                        self._bkg_model_counts_sgl_psd[n,v][self._sgl_mask])

                if np.sum(~self._sgl_mask)>0:
                    
                    for v, d in enumerate(self._sgl_dets_to_use):

                        loglike += self._likelihood.Cash_stat(
                            self._active_time_counts_energy_psd_dict[n,d][~self._sgl_mask],
                            self._expected_model_counts_psd[n,v][~self._sgl_mask]+
                            self._bkg_model_counts_psd[n,v][~self._sgl_mask])

            if 'double' in self._event_types:
                for v, d in enumerate(self._me2_dets_to_use):

                    loglike += self._likelihood.Cash_stat(
                        self._active_time_counts_energy_me2_dict[n,d],
                        self._expected_model_counts_me2[n,v]+
                        self._bkg_model_counts_me2[n,v])

            if 'triple' in self._event_types:
                for v, d in enumerate(self._me3_dets_to_use):

                    loglike += self._likelihood.Cash_stat(
                        self._active_time_counts_energy_me3_dict[n,d],
                        self._expected_model_counts_me3[n,v]+
                        self._bkg_model_counts_me3[n,v])

        return loglike
