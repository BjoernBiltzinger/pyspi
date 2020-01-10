from threeML import PluginPrototype
from threeML.io.file_utils import sanitize_filename
try:
    from pyspi.spi_analysis import *
except:
    raise ImportError('You need to have pyspi installed to use this plugin!')



__instrument_name = "INTEGRAL SPI (with PySPI)"

class SPILike(PluginPrototype):
    """
    Plugin for the data of SPI, based on PySPI
    """
    def __init__(self, name, pyspi_config):
        """
        :param name: name of instance
        :param pyspi_config: YAML config file
        """
        # There are no nuisance parameters

        nuisance_parameters = {}

        super(SPILike, self).__init__(name, nuisance_parameters=nuisance_parameters)

        if not isinstance(pyspi_config, dict):

            # Assume this is a file name
            configuration_file = sanitize_filename(pyspi_config)

            assert os.path.exists(pyspi_config), "Configuration file %s does not exist" % configuration_file

            # Read the configuration
            with open(configuration_file) as f:

                self._configuration = yaml.safe_load(f)

        else:

            # Configuration is a dictionary. Nothing to do
            self._configuration = pyspi_config


    def set_model(self, likelihood_model):
        """
        Set the model to be used in the joint minimization.
        :param likelihood_model: likelihood model instance
        """

        self._likelihood_model = likelihood_model

        self._spi_analysis = SPIAnalysis(self._configuration, self._likelihood_model)
        #self._gta, self._pts_energies = _get_PySpi_instance(self._configuration, likelihood_model_instance)

    def _update_model_in_pyspi(self):
        """
        Update model in pyspi
        """
        self._spi_analysis.update_model(self._likelihood_model)
        
    def get_log_like(self):
        """
        Return log likelihood for current parameters stored in the ModelManager instance
        """

        # Update all sources on the fermipy side
        #self._update_model_in_pyspi()

        # Get value of the log likelihood
        return self._spi_analysis.get_log_like(self._likelihood_model)

    def inner_fit(self):
        """
        This is used for the profile likelihood. Keeping fixed all parameters in the
        LikelihoodModel, this method minimize the logLike over the remaining nuisance
        parameters, i.e., the parameters belonging only to the model for this
        particular detector. If there are no nuisance parameters, simply return the
        logLike value.
        """
        return self.get_log_like()

    def view_lightcurve(self):
        """
        Wrapper to view lightcurve of spi_analysis object
        :return: figure
        """
        return self._spi_analysis.plot_lightcurve()
    
def _get_PySpi_instance(configuration, likelihood_model):
    """
    Generates a 'model' configuration for pyspi starting from a likelihood_model from astromodels

    :param configuration: Dictionary with configurations
    :param likelihood_model: input astromodels likelihood_model
    :return: ????????
    """

    # Now iterate over all sources contained in the likelihood model
    sources = []

    # point sources
    for point_source in likelihood_model.point_sources.values():  # type: astromodels.PointSource

        this_source = {}
        
        spectal_parameters = {}
        for i, component in enumerate(point_source._components.values()):

            spectral_parameters_component = {}
            for key in component.shape.parameters.keys():
                spectral_parameters_component[key] = component.shape.parameters[key].value

            spectal_parameters[i] = spectral_parameters_component

        this_source['spectal_parameters'] = spectal_parameters
        this_source['name'] = point_source.name
        this_source['ra'] = point_source.position.ra.value
        this_source['dec'] = point_source.position.dec.value
        this_source['response'] = None
        this_source['predicted_count_rates'] = None
        # The spectrum used here is unconsequential, as it will be substituted by a FileFunction
        # later on. So I will just use PowerLaw for everything

        sources.append(this_source)

    configuration['point_sources'] = sources 
    
    spi_analysis = SPIAnalysis(configuration)

    return spi_analysis
