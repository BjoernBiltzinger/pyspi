from pyspi.spi_grb_analysis import GRBAnalysisRMF, GRBAnalysisPhotopeak
from pyspi.spi_constantsource_analysis import SPI_CS_Analysis

def getspianalysis(configuration, likelihood_model, photopeak_only=False):
    """
    Init a SPIAnalysis object that is used to handle the spi model evaluation during
    the fit
    :param configuration: Configuration dictionary
    :param likelihood_model: The inital astromodels likelihood_model
    """
    # Which kind of analysis?
    analysis = configuration['Special_analysis']
    if analysis=='GRB':
        if photopeak_only:
            return GRBAnalysisPhotopeak(configuration, likelihood_model)
        else:
            return GRBAnalysisRMF(configuration, likelihood_model)
    elif analysis=='Constant_Source':
        return SPI_CS_Analysis(configuration, likelihood_model)
    else:
        raise AssertionError('Please use a valid Special Analysis type.' \
                             ' Either GRB for a GRB analysis or Constant_Source'/
                             ' for constant sources (like point sources that are constant in time)')
            
