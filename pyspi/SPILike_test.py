from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from astromodels.functions.priors import Uniform_prior, Cosine_Prior
from SPI_DRM import SPI_DRM

class SPILike_test(DispersionSpectrumLike):
    def __init__(self,
                 name,
                 observation,
                 drm_generator=None,
                 background=None,
                 det=0,
                 free_position=True,
                 verbose=True,
                 ra_start=5.,
                 dec_start=5.):
        """
        BALROGLike is a general plugin for fitting GBM spectra and locations at the same time


        :param name: plugin name
        :param observation: observed spectrum
        :param drm_generator: the drm generator for this 
        :param background: background spectrum
        :param time: time of the observation
        :param free_position: keep the position free
        :param verbose: the verbosity level of the plugin
        """

        self._free_position = free_position

        self._det = det
        spi_drm =  SPI_DRM(drm_generator, ra_start, dec_start, det=det)
        self._ra_s = ra_start
        self._dec_s = dec_start
        observation._rsp = spi_drm
        


        super(SPILike_test, self).__init__(name, observation, background,
                                         verbose)

        # only on the start up

        #self._rsp.set_time(time)

    def set_model(self, likelihoodModel):
        """
        Set the model and free the location parameters


        :param likelihoodModel:
        :return: None
        """

        # set the standard likelihood model

        super(SPILike_test, self).set_model(likelihoodModel)

        # now free the position
        # if it is needed

        if self._free_position:

            if self._verbose:
                print('Freeing the position of %s and setting priors' % self.name)

            for key in self._like_model.point_sources.keys():
                self._like_model.point_sources[key].position.ra.free = True
                self._like_model.point_sources[key].position.dec.free = True

                self._like_model.point_sources[
                    key].position.ra.prior = Uniform_prior(
                    lower_bound=0., upper_bound=40)
                self._like_model.point_sources[
                    key].position.dec.prior = Cosine_Prior(
                    lower_bound=-20., upper_bound=20)

        self._rsp.set_location(self._ra_s, self._dec_s)

    def get_model(self):

        # Here we update the GBM drm parameters which creates and new DRM for that location
        # we should only be dealing with one source for GBM

        # update the location

        if self._free_position:

            # assumes that the is only one point source which is how it should be!
            ra, dec = self._like_model.get_point_source_position(0)
 
            self._rsp.set_location(ra, dec)
            
        return super(SPILike_test, self).get_model()

    @classmethod
    def from_spectrumlike(cls, spectrum_like, det=0, drm_generator=None,free_position=True, ra_start=5., dec_start=5.):
        """
        Generate a BALROGlike from an existing SpectrumLike child
        
        
        :param spectrum_like: the existing spectrumlike
        :param time: the time to generate the RSPs at
        :param drm_generator: optional BALROG DRM generator
        :param free_position: if the position should be free
        :return: 
        """

        return cls(spectrum_like.name, spectrum_like._observed_spectrum,drm_generator,
                   spectrum_like._background_spectrum, det, free_position,
                   spectrum_like._verbose, ra_start, dec_start)
