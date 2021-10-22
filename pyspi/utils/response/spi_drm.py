import numpy as np
from threeML.utils.OGIP.response import InstrumentResponse


class SPIDRM(InstrumentResponse):
    def __init__(self, drm_generator, ra, dec):
        """
        Init a SPIDRM object which is based on the InstrumenResponse
        class from 3ML. Contains everything that is necessary for
        3ML to recognize it as a response.
        :param drm_generator: DRM generator for the SPI Response
        :param ra: ra of source (in ICRS)
        :param dec: dec of source (in ICRS)
        :return: Object
        """
        self._drm_generator = drm_generator

        self._drm_generator.set_location(ra, dec)
        self._min_dist = np.deg2rad(.5)

        self._ra = ra
        self._dec = dec

        super(SPIDRM, self).__init__(
            self._drm_generator.matrix,
            self._drm_generator.ebounds,
            self._drm_generator.monte_carlo_energies,
        )

    def set_location(self, ra, dec, cache=False):
        """
        Set the source location
        :param ra: ra of source (in ICRS)
        :param dec: dec of source (in ICRS)
        :return:
        """
        self._drm_generator.set_location(ra, dec)

        self._ra = ra
        self._dec = dec

        self._matrix = self._drm_generator.matrix
        self._matrix_transpose = self._matrix.T

    def set_location_direct_sat_coord(self, azimuth, zenith, cache=False):
        """
        Set the source location
        :param azimuth: az poisition in the sat. frame
        :param zenith: zenith poisition in the sat. frame
        :return:
        """
        self._ra, self._dec =\
            self._drm_generator.set_location_direct_sat_coord(azimuth, zenith)

        self._matrix = self._drm_generator.matrix
        self._matrix_transpose = self._matrix.T

    def clone(self) -> "SPIDRM":
        """
        return a new response with the contents of this response
        :returns: new cloned response
        """

        # We have to be carefull to not clone the irf_read_object.
        # Because it is huge! ~ 1GB
        new_drm_generator = self._drm_generator.clone()

        #    ResponseRMFNew(monte_carlo_energies=self._drm_generator.monte_carlo_energies,
        #                   ebounds=self._drm_generator.ebounds,
        #                   response_irf_read_object=self._drm_generator._irf_ob,
        #                   sc_matrix=self._drm_generator._sc_matrix,
        #                   det=self._drm_generator._det,
        #                   fixed_rsp_matrix=self._drm_generator._rsp_matrix)
        return SPIDRM(drm_generator=new_drm_generator,
                      ra=self._ra,
                      dec=self._dec,
        )
