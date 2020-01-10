class SPI_CS_Analysis(object):

    def __init__(self, configuration, likelihood_model):
        """
        Init a Spi Analysis object for an analysis of a constant source 
        (Point sources and extended sources). Superclass of SPIAnalysis.
        :param configuration: Configuration dictionary
        :param likelihood_model: The inital astromodels likelihood_model
        """
        raise NotImplementedError('Constant source analysis not implemented at the moment!') 

