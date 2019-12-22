from pyspi.spi_data import *
from pyspi.spi_data_grb_synth import *
from pyspi.spi_response import *
from pyspi.spi_pointing import *
from pyspi.spi_frame import *
import astropy.units as u
from astropy.coordinates import ICRS, Galactic, SkyCoord


from threeML.utils.statistics.likelihood_functions import *
from threeML import *
class SPIAnalysis(object):

    def __init__(self, configuration, likelihood_model):
        """
        Init a SPIAnalysis object that is used to handle the spi model evaluation during
        the fit
        :param configuration: Configuration dictionary
        :param likelihood_model: The inital astromodels likelihood_model
        """

        self._emin = float(configuration['emin'])
        self._emax = float(configuration['emax'])

        #self._dets = configuration['Detectors_to_use']
        self._event_types = configuration['Event_types']

        if configuration['Special_analysis']=='GRB':
            self._analysis = 'GRB'
            self._time_of_grb = configuration['Time_of_GRB_UTC']
            self._active_time = configuration['Active_Time']
            self._bkg_time_1 = configuration['Background_time_interval_1']
            self._bkg_time_2 = configuration['Background_time_interval_2']
            self._simmulate = configuration['Simmulate']

            self._get_start_and_stop_times()

        else:
            raise NotImplementedError('Only GRB analysis is implemented at the moment!')

        self._dets = configuration['Detectors_to_use']

        self._ebounds = np.logspace(np.log10(self._emin), np.log10(self._emax), 30)

        self._init_response()
        
        self._init_data(self._simmulate)

        self.get_background_estimation()

        self._init_frame()
        
        self.set_model(likelihood_model)
        
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
        
    def _init_response(self):
        """
        Initalize the response object
        :return:
        """
        self._response_object = SPIResponse(ebounds=self._ebounds)

        #if self._analysis == 'GRB':
        #    # Dummy response - needed for later timeseries building
        #    self._response_dummy =  self._response_object.get_binned_effective_area_det_trapz(0.2,0.2,0)

        
    def _init_data(self, simmulate):
        """
        Get the data object with all the data we need
        :return:
        """
        if self._analysis=='GRB':
            if not simmulate:
                self._data_object = SpiData_GRB(self._time_of_grb, afs=True, ebounds=self._ebounds)
                
                # Bin the data in energy and time - dummy values for time bin step
                # size and ebounds
                self._data_object.time_and_energy_bin_sgl(time_bin_step=1,
                                                          start=self._bkg1_start-10,
                                                          stop=self._bkg2_stop+10)
                
            else:
                def GRB_spectrum(E):
                    return 0.01*np.power(float(E)/100., -2.)
                
                self._data_object = SpiData_synthGRB(self._time_of_grb, self._response_object,
                                                     ra=5., dec=5., duration_of_GRB=30.,
                                                     GRB_spectrum_function=GRB_spectrum,
                                                     afs=True, ebounds=self._ebounds,
                                                     start=self._bkg1_start-10,
                                                     stop=self._bkg2_stop+10)
                
            if 'single' in self._event_types:
                self._sgl_dets = self._data_object.energy_sgl_dict.keys()
                
                # Build a mask to cover the time bins of the active time
                time_bins = self._data_object.time_bins
                
                self._active_time_mask = np.logical_and(time_bins[:,0]>self._active_start,
                                      time_bins[:,1]<self._active_stop)

                # Get time bins
                self._active_time_bins = self._data_object.time_bins[self._active_time_mask]
                                
                self._active_time_counts_energy_sgl_dict = {}
                for d in self._data_object.energy_sgl_dict.keys():
                    self._active_time_counts_energy_sgl_dict[d] = np.sum(self._data_object.energy_and_time_bin_sgl_dict[d][self._active_time_mask], axis=0)

                self._real_start_active = self._active_time_bins[0,0]
                self._real_stop_active = self._active_time_bins[-1,-1]

                    
            else:
                raise NotImplementedError('Only single events implemented!')
            
            # Add other det types

        
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

        assert name not in self._point_sources.keys(), 'Can not create the source {} twice!'.format(name)
        # ra and dec to sat coord
        icrscoord = SkyCoord(ra=point_source.position.ra.value, dec=point_source.position.dec.value, unit='deg', frame='icrs')
        satcoord = icrscoord.transform_to(self._frame_object)

        ra_sat = satcoord.lon.deg
        dec_sat = satcoord.lat.deg 
        
        # Calculate responses for all dets
        response = {}
        if 'single' in self._event_types:
            for d in self._sgl_dets:
                response[d] = self._response_object.set_location(ra_sat, dec_sat, d, trapz=True)
                
        # Get current spectrum of source
        spectrum_bins = self.calculate_spectrum(point_source)

        predicted_count_rates = {}
        # Get the predicted count rates in all dets in all PHA bins (individual for all pointings later)
        if 'single' in self._event_types:
            predicted_count_rates[d] = np.multiply(response[d], spectrum_bins)
        
        spectal_parameters = {}
        
        for i, component in enumerate(point_source._components.values()):

            spectral_parameters_component = {}
            for key in component.shape.parameters.keys():
                spectral_parameters_component[key] = component.shape.parameters[key].value

            spectal_parameters[i] = spectral_parameters_component
                
        # Create the entry of the point source
        self._point_sources[name] = {'ra': point_source.position.ra.value, 'dec': point_source.position.dec.value,
                                     'response': response, 'predicted_count_rates': predicted_count_rates,
                                     'spectal_parameters': spectal_parameters}

            
    def update_pointsource(self, name, point_source):
        """
        Update the influence of a point source on all dets
        :param name: Name of point source
        :param point_source: Astromodel point source 
        :return:
        """

        assert name in self._point_sources.keys(), 'The source with the name {} does not exists yet. We can not create a new source on the fly!'.format(name)
        
        # if position has changed recalculate response with new position - response is a callable function of Energy
        if point_source.position.ra.value != self._point_sources[name]['ra'] or point_source.position.dec.value != self._point_sources[name]['dec']:
            # ra and dec to sat coord
            icrscoord = SkyCoord(ra=point_source.position.ra.value, dec=point_source.position.dec.value, unit='deg', frame='icrs')

            sep = icrscoord.separation(self._pointing_icrs).deg

            if sep<45:
                satcoord = icrscoord.transform_to(self._frame_object)

                ra_sat = satcoord.lon.deg
                dec_sat = satcoord.lat.deg 

                # Calculate responses for all dets
                response = {}
                if 'single' in self._event_types:
                    for d in self._sgl_dets:
                        response[d] = self._response_object.set_location(ra_sat, dec_sat, d, trapz=True)

            # When sep>45 it is outside of FOV -> set responses to zero - TODO: use good prior to
            # avoid this
            else:
                response = {}
                if 'single' in self._event_types:
                    for d in self._sgl_dets:
                        response[d] = np.zeros_like(self._ebounds[:-1])
        else:
            response = self._point_sources[name]['response']
        
        # Get current spectrum of source
        spectrum_bins = self.calculate_spectrum(point_source)

        predicted_count_rates = {}
        # Get the predicted count rates in all dets in all PHA bins (individual for all pointings later)
        if 'single' in self._event_types:
            for d in self._sgl_dets:
                predicted_count_rates[d] = np.multiply(response[d], spectrum_bins)

        spectal_parameters = {}
        for i, component in enumerate(point_source._components.values()):

            spectral_parameters_component = {}
            for key in component.shape.parameters.keys():
                spectral_parameters_component[key] = component.shape.parameters[key].value

            spectal_parameters[i] = spectral_parameters_component


        # Update the entry of the point source
        self._point_sources[name] = {'ra': point_source.position.ra.value,
                                     'dec': point_source.position.dec.value,
                                     'response': response,
                                     'predicted_count_rates': predicted_count_rates,
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
        if self._analysis == 'GRB':
            # Initalize the polynominal background fit to the time before and after the trigger time
            # to get the temporal changing Background
            self._bkg = {}
            self._bkg_active = {}
            if 'single' in self._event_types:
                
                for d in self._sgl_dets:

                    # Use threeML times series to calculate the background polynominals for every det
                    # and det PHA channel
                    
                    tsb = TimeSeriesBuilder.from_spi_spectrum('spi_det{}'.format(d),
                                                              self._data_object,
                                                              d,
                                                              self._response_object)
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

                    self._bkg[d] = {'error': bkg_error, 'counts': bkg_counts} 
                    self._bkg_active[d] = {'error_active': bkg_error_active, 'counts_active': bkg_counts_active}  
        else:
            raise NotImplementedError('Only GRB analysis implemented at the moment!')

    def _get_model_counts(self, likelihood_model):
        """
        Get the current model counts for all dets
        :param likelihood_model: current likelihood_model
        :return:
        """

        self.update_model(likelihood_model)
        expected_model_counts = np.zeros((len(self._sgl_dets),len(self._data_object.ene_min)))
        for i, d in enumerate(self._sgl_dets):
            for point_s in self._point_sources.keys():
                expected_model_counts[i] += self._point_sources[point_s]['predicted_count_rates'][d]

        self._expected_model_counts = expected_model_counts
        
    def get_log_like(self, likelihood_model):
        """
        Return the current log likelihood value, only point_source at the moment
        :return:
        """
        
        self._get_model_counts(likelihood_model)

        if self._analysis=='GRB':
            # Use pgstat likelihood (background gaussian + signal poisson)
            loglike = 0
            if 'single' in self._event_types:
                for v, d in enumerate(self._sgl_dets):
                    for i in range(len(self._expected_model_counts)):

                        loglike += poisson_observed_gaussian_background(
                            np.array([self._active_time_counts_energy_sgl_dict[d][i]]),
                            np.array([self._bkg_active[d]['counts_active'][i]]),
                            np.array([self._bkg_active[d]['error_active'][i]]),
                            np.array([self._expected_model_counts[v][i]]))[0]


            return loglike
        else:
            raise NotImplementedError('Only GRB analysis at the moment!')

    def plot_lightcurve(self):
        """
        Plot data and bkg fits of sgl dets
        :return: fig
        """
        assert self._analysis == 'GRB', 'Only possible for GRB analysis'

        n_dets = len(self._sgl_dets)

        n_rows = np.ceil(n_dets/2.)
        n_colums = 2
        
        fig, axes = plt.subplots(n_dets, 1, sharex=True, figsize=(8.27, 11.69))
        
        for i in range(n_dets):
            # Plot data
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._data_object.energy_and_time_bin_sgl_dict[self._sgl_dets[i]], axis=1)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='black')

            # Plot the bkg fit
            axes[i].plot(np.mean(self._data_object.time_bins, axis=1), np.sum(self._bkg[self._sgl_dets[i]]['counts'], axis=0)/(self._data_object.time_bins[:,1]-self._data_object.time_bins[:,0]), color='red')

            # Plot active time
            axes[i].axvspan(self._active_start, self._active_stop, color='green', alpha=0.2)

            # Plot bkg times
            axes[i].axvspan(self._bkg1_start, self._bkg1_stop, color='red', alpha=0.2)
            axes[i].axvspan(self._bkg2_start, self._bkg2_stop, color='red', alpha=0.2)

            axes[i].locator_params(axis='y', nbins=1)
        axes[-1].set_xlabel('Time [s]')
        axes[-1].set_xlim(self._bkg1_start, self._bkg2_stop)

        ax_frame = fig.add_subplot(111, frameon=False)
        ax_frame.tick_params(labelcolor='none', top='off', bottom='off', left='off', right='off')
        ax_frame.set_ylabel('Count rates [cnts s$^{-1}$]')

        fig.tight_layout()
        fig.subplots_adjust(hspace=0)
        
        return fig

    def _loadtxt2d(self, intext):
        try:
            return np.loadtxt(intext, ndmin=2)
        except:
            return np.loadtxt(intext)

    
    def plot_fit_data(self, post_equal_weights_file, likelihood_model):
        """
        Plot the data and the fit result in one plot
        :param post_equal_weights_file: file with post_equal_weights parameter samples - for example from multinest
        :return: fig
        """

        if self._analysis=='GRB':
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
            model_counts = np.empty((n_ppc, len(self._sgl_dets), len(self._ebounds)-1))
            for i in range(n_ppc):
                likelihood_model.set_free_parameters(masked_parameter_samples[i])
                self._get_model_counts(likelihood_model)
                model_counts[i] = self._expected_model_counts
                
            # poisson noise
            model_counts = np.random.poisson(model_counts)

            # BKG wit gaussian noise
            bkg_counts = np.empty((n_ppc, len(self._sgl_dets), len(self._ebounds)-1))
            
            for j, d in enumerate(self._sgl_dets):
                bkg_counts[:,j,:] = np.random.normal(self._bkg_active[d]['counts_active'], self._bkg_active[d]['error_active'], size=(n_ppc,len(self._ebounds)-1))

            
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
                    if j in self._sgl_dets:
                        
                        # Data rate
                        active_data = self._active_time_counts_energy_sgl_dict[d]/total_active_time#or axis=1
                        # BKG rate
                        bkg_rate = np.sum(self._bkg[d]['counts'])/total_active_time
                        # PPC fit count spectrum

                        # get counts for all sample parameters and the likelihood_model
                        model_bkg_rates_det = (model_counts[:,index,:]+bkg_counts[:,index,:])/total_active_time #? Where bkg?
                        
                        # Plot number of this det
                        #plot_number = ((j)*4)/19
                        #if plot_number<2:
                        #    x_plot_number = 0
                        #    y_plot_number = plot_number
                        #else:
                        #    x_plot_number = 1
                        #    y_plot_number = plot_number-2
                        
                        q_levels = [0.68,0.95]
                        colors = ['lightgreen', 'darkgreen']# TODO change this to more fancy colors
                        
                        # get 68 and 95 % boundaries and plot them
                        for i,level in enumerate(q_levels):
                            low = np.percentile(model_bkg_rates_det, 50-50*level, axis=0)
                            high = np.percentile(model_bkg_rates_det, 50+50*level, axis=0)
                            axes_array[plot_number].fill_between(self._ebounds[1:],
                                                           low,
                                                           high,
                                                           color=colors[i],
                                                           alpha=0.5,
                                                           zorder=i+1,
                                                           step='post')
                        
                        axes_array[plot_number].step(self._ebounds[1:],
                                               active_data,
                                               where='post',
                                               color='black',
                                               label='Detector {}'.format(j))
                        if (plot_number/float(ncol)).is_integer():
                            axes_array[plot_number].set_ylabel('Count rate [cts s$^-1$]')
                        
                        index+=1
                    # If det not used only plot legend entry with remark "not used or defect"
                    else:
                        # Plot number of this det
                        #plot_number = (j*4)/19

                        #if plot_number<2:
                        #    x_plot_number = 0
                        #    y_plot_number = plot_number
                        #else:
                        #    x_plot_number = 1
                        #    y_plot_number = plot_number-2


                        
                        red_indices.append(j)
                        axes_array[plot_number].plot([], [], ' ', label='Detector {} \n Not used'.format(j))

                # Make legend and mark the not used dets red
                for i, ax in enumerate(axes.flatten()):
                    ax.set_xlabel('Energy [keV]')
                    l = ax.legend()
                    if i in red_indices:
                        l.get_texts()[0].set_color("red")
                    ax.set_xscale('log')
                    #for n in np.arange(5):
                    #    if i*5+n<19:
                    #        if i*5+n in red_indices:
                    #            l.get_texts()[n].set_color("red")
                fig.tight_layout()
                fig.subplots_adjust(hspace=0, wspace=0) 
                fig.savefig('data_plot.pdf')
        else:
            raise NotImplementedError('Only implemented for GRBs at the moment!')
