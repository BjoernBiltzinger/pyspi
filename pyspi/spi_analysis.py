from pyspi.spi_grb_analysis import GRBAnalysisRMF, GRBAnalysisPhotopeak
from pyspi.spi_constantsource_analysis import ConstantSourceAnalysisRMF, ConstantSourceAnalysisPhotopeak

def SpiAnalysisList(pyspi_config):#, likelihood_model):
    """
    Init a SPIAnalysis object that is used to handle the spi model evaluation during
    the fit
    :param configuration: Configuration dictionary
    :param likelihood_model: The inital astromodels likelihood_model
    """

    if not isinstance(pyspi_config, dict):

        if isinstance(pyspi_config, Config):
            configuration = pyspi_config
        else:
            # Assume this is a file name
            configuration_file = sanitize_filename(pyspi_config)

            assert os.path.exists(pyspi_config), "Configuration file %s does not exist" % configuration_file

            # Read the configuration
            with open(configuration_file) as f:

                configuration = yaml.safe_load(f)

    else:

        # Configuration is a dictionary. Nothing to do
        configuration = pyspi_config

    # Which kind of analysis?
    analysis = configuration['Special_analysis']
    photopeak_only = configuration['Use_only_photopeak']
    print(photopeak_only)

    if analysis=='GRB':

        if photopeak_only:
            analysis_class = GRBAnalysisPhotopeak

        else:
            analysis_class = GRBAnalysisRMF

    elif analysis=='Constant_Source':

        if photopeak_only:
            analysis_class = ConstantSourceAnalysisPhotopeak

        else:
            analysis_class = ConstantSourceAnalysisRMF

    else:
        raise AssertionError('Please use a valid Special Analysis type.' \
                             ' Either GRB for a GRB analysis or Constant_Source'/
                             ' for constant sources (like point sources that are constant in time)')

    # Which detectors should be used? Can be the ids for single, double or tripple detectors.
    dets = np.array(configuration['Detectors_to_use'])

    analysis_list = []

    for d in dets:
        analysis_list.append(analysis_class(configuration, d))

        
    return analysis_list
