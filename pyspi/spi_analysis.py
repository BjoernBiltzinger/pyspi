from pyspi.spi_grb_analysis import SPI_GRB_Analysis
from pyspi.spi_constantsource_analysis import SPI_CS_Analysis

class SPIAnalysis(SPI_GRB_Analysis, SPI_CS_Analysis):

    def __init__(self, configuration, likelihood_model):
        """
        Init a SPIAnalysis object that is used to handle the spi model evaluation during
        the fit
        :param configuration: Configuration dictionary
        :param likelihood_model: The inital astromodels likelihood_model
        """

        # Which kind of analysis?
        self._analysis = configuration['Special_analysis']

        if self._analysis=='GRB':
            SPI_GRB_Analysis.__init__(self, configuration, likelihood_model)

        elif self._analysis=='Constant_Source':
            SPI_CS_Analysis.__init__(self, configuration, likelihood_model)

        else:
            raise AssertionError('Please use a valid Special Analysis type. Either GRB for a GRB analysis or Constant_Source for constant sources (like point sources that are constant in time)')
            
