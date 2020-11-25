from threeML import PluginPrototype
from threeML.io.file_utils import sanitize_filename
from astromodels import Parameter
import collections
from threeML import *
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from astromodels import *
from threeML.plugins.SpectrumLike import SpectrumLike

"""
TODO List
add set_active_time
add set_background_time

add view_lightcurves()
"""

__instrument_name = "INTEGRAL SPI (with PySPI)"

# One plugin for Photopeak, one for full response (with Dispersion)
class SPILikeGRBPhotopeak(SpectrumLike):
    """
    Plugin for the data of SPI, based on PySPI
    """
    def __init__(
            self,
            name,
            observation,
            background=None,
            free_position=False,
            verbose=True,
            rsp_object=None,
            **kwargs
    ):
        """
        :param name: name of instance
        :param pyspi_setup: Pyspi Setup object
        """
        self._free_position = free_position
        super(SPILikeGRBPhotopeak, self).__init__(name,
                                                  observation,
                                                  background,
                                                  verbose,
                                                  **kwargs)
        # Construct the "response" object. This Plugin only uses the photopeak effective area,
        # therefore the response gives a vector of effective areas at the ebounds of the bins.
        # This vector is multiplied with the flux counts in the same bins.
        # In this apporach input_bins=output_bins.

        self._rsp = rsp_object
        
        # We have one special feature in SPI that is the "electronic noise" range from 1400 keV to 1700 keV
        # In this range the non-psd single events suffer from some unknown problem which makes the single
        # count rates unreliable in this region. The PSD single events do not have this problem.
        # Therefore only the psd events are used in this "electronic range" region, whereas in the rest
        # non-psd and psd events are used together. To account for the droping of the non-psd events in this
        # energy range the response has to be adjusted to the fraction of single events that do pass the psd.
        # This is modeled with the psd_eff nuiscance parameter that should be around ~85% in this energy region
        # but can vary slightly. This effect is only important if this plugin is for a single detector
        # and the energy range between 1400-1700 keV is in the analysis.

    def _evaluate_model(self, true_fluxes=None):
        """
        Evaluate model by multiplying the average effective area of all Ebins with
        the flux in the corresponding incoming Ebin.

        This is photopeak analysis specific! For the general case with energy dispersion, this function needs to
        be changed.

        :param true_fluxes: Flux of incoming spectrum; if None than the flux is computed with the current
        parameters
        """
        if true_fluxes is None:
            true_fluxes = self._integral_flux(self._observed_spectrum.bin_stack[:,0],
                                              self._observed_spectrum.bin_stack[:,1])
        if np.any(self._psd_mask):
            self._psd_eff_area[self._psd_mask] = self._like_model.psd_eff_spi.value
        return self._psd_eff_area*self._rsp.effective_area*true_fluxes

    def set_model(self, likelihood_model):
        """
        Set the model to be used in the joint minimization.
        :param likelihood_model: likelihood model instance
        """

        super(SPILikeGRBPhotopeak, self).set_model(likelihood_model)

        if self._free_position:
            print("Freeing the position of %s and setting priors" % self.name)
            for key in self._like_model.point_sources.keys():
                self._like_model.point_sources[key].position.ra.free = True
                self._like_model.point_sources[key].position.dec.free = True

                self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                    lower_bound=0.0, upper_bound=360
                )
                self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                    lower_bound=-90.0, upper_bound=90
                )

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value
        else:
            for key in self._like_model.point_sources.keys():

                self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                    lower_bound=0.0, upper_bound=360
                )
                self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                    lower_bound=-90.0, upper_bound=90
                )

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector

        # We have one special feature in SPI that is the "electronic noise" range from 1400 keV to 1700 keV
        # In this range the non-psd single events suffer from some unknown problem which makes the single
        # count rates unreliable in this region. The PSD single events do not have this problem.
        # Therefore only the psd events are used in this "electronic range" region, whereas in the rest
        # non-psd and psd events are used together. To account for the droping of the non-psd events in this
        # energy range the response has to be adjusted to the fraction of single events that do pass the psd.
        # This is modeled with the psd_eff nuiscance parameter that should be around ~85% in this energy region
        # but can vary slightly. This effect is only important if this plugin is for a single detector
        # and the energy range between 1400-1700 keV is in the analysis.
        self._psd_mask = np.zeros(len(self._rsp._ebounds)-1, dtype=bool)
        self._psd_eff_area = np.ones(len(self._rsp._ebounds)-1)
        if self._rsp._det in range(19):
            # If several spi detector plugins are created this will overwrite each time.
            # We only need this once (at least per sw...)
            # TODO: When to fit a new psd_eff? Every sw? Every orbit? Every year?
            self._psd_mask = self._rsp._psd_bins
            if np.any(self._rsp._psd_bins):
                assert "psd_eff_spi" in self._like_model.parameters.keys(), "Need the psd_spi_eff parameter in the model!"
                self._psd_eff_area[self._rsp._psd_bins] = self._like_model.psd_eff_spi.value*np.ones(np.sum(self._rsp._psd_bins))

                #print(self._psd_eff_area)

    def get_model(self, true_fluxes=None):

        if self._free_position:

            # assumes that the is only one point source which is how it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector

        return super(SPILikeGRBPhotopeak, self).get_model(true_fluxes=true_fluxes)


    @classmethod
    def from_spectrumlike(
            cls, spectrum_like, rsp_object, free_position=False
    ):
        """
        Generate SPILikeGRB from an existing SpectrumLike child
        """
        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            spectrum_like._background_spectrum,
            free_position,
            spectrum_like._verbose,
            rsp_object
        )
