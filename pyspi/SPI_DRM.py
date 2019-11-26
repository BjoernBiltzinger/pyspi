from threeML.utils.OGIP.response import InstrumentResponse




class SPI_DRM(InstrumentResponse):

    def __init__(self, spi_response, ra, dec, det):
        """
        
        :param drm_generator: BALROG DRM generator
        :param ra: RA of the source
        :param dec: DEC of the source
        """


        self._drm_generator = spi_response
        self._det=det
        self._drm_generator.set_location(ra,dec, det)

        super(SPI_DRM, self).__init__(self._drm_generator.matrix,
                                         self._drm_generator.ebounds,
                                         self._drm_generator.ebounds)

    def set_location(self, ra, dec):
        """
        Set the source location
        :param ra: 
        :param dec: 
        :return: 
        """

        self._drm_generator.set_location(ra, dec, self._det)

        self._matrix = self._drm_generator.matrix
