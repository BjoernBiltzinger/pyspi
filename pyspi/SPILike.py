import collections

import numpy as np
from astromodels import Parameter, Model
from astromodels.functions.priors import Cosine_Prior, Uniform_prior
from threeML import PluginPrototype
from threeML.io.file_utils import sanitize_filename
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.io.logging import setup_logger
from pyspi.utils.response.spi_response import SPIDRM
from typing import Optional

"""
TODO List
add view_lightcurves()
"""

__instrument_name = "INTEGRAL SPI (with PySPI)"


log = setup_logger(__name__)

class SPILike(DispersionSpectrumLike):
    """
    Plugin for the data of SPI, based on PySPI
    """
    def __init__(
            self,
            name: str,
            observation,
            background,
            bkg_base_array,
            free_position: bool,
            verbose:bool=True,
            **kwargs
    ):
        """

        """
        self._free_position: bool = free_position

        if not isinstance(
                observation.response, SPIDRM
            ):


            log.error("The response associated with the observation is not a SPIDRM")

            raise AssertionError()

        super(SPILike, self).__init__(name,
                                      observation,
                                         background,
                                         verbose,
                                         **kwargs)
        #self._add_bkg_nuisance_parameter(bkg_parameters)
        self._bkg_base_array = bkg_base_array
        self._bkg_array = np.ones(len(self._bkg_base_array))

    def set_model(self, likelihood_model: Model) -> None:
        """
        Set the model to be used in the joint minimization.
        :param likelihood_model: likelihood model instance
        """

        super(SPILike, self).set_model(likelihood_model)

        if self._free_position:
            log.info(f"Freeing the position of {self.name} and setting priors")
            
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

                #self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                #    lower_bound=0.0, upper_bound=360
                #)
                #self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                #    lower_bound=-90.0, upper_bound=90
                #)

                ra: float = self._like_model.point_sources[key].position.ra.value
                dec: float = self._like_model.point_sources[key].position.dec.value

        self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector
        """
        TODO need this psd mask stuff later
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
        """

    def _evaluate_model(self, precalc_fluxes=None):

        source = super(SPILike, self)._evaluate_model(precalc_fluxes=precalc_fluxes)
        self._update_bkg_array()
        bkg = self._bkg_array*self._bkg_base_array
        return source+bkg

    def get_model(self, precalc_fluxes: Optional[np.ndarray]=None) -> np.ndarray:

        if self._free_position:

            # assumes that the is only one point source which is how it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector

        return super(SPILike, self).get_model(precalc_fluxes=precalc_fluxes)
        #source = super(SPILike, self).get_model(precalc_fluxes=precalc_fluxes)
        #self._update_bkg_array()
        #bkg = self._bkg_array*self._bkg_base_array
        #return source+bkg

    def _add_bkg_nuisance_parameter(self, bkg_parameters) -> None:
        """TODO describe function

        :param bkg_parameters: 
        :type bkg_parameters: 
        :returns: 

        """
        
        self._bkg_parameters = bkg_parameters
        for parameter in bkg_parameters:
            self.nuisance_parameters[parameter.name] = parameter
        self._bkg_array = np.ones(len(bkg_parameters))

    def _update_bkg_array(self) -> None:
        """TODO describe function

        :returns: 

        """
        
        for key in self._like_model.parameters.keys():
            if "bkg" in key:
                idx = int(key.split("_")[-1])
                self._bkg_array[idx] = self._like_model.parameters[key].value

    @classmethod
    def from_spectrumlike(
        cls,
        spectrum_like,
        bkg_base_array,
        #bkg_parameters,
        free_position=False
    ):
        """
        Generate SPILikeGRB from an existing SpectrumLike child
        :param spectrum_like: SpectrumLike child
        :param rsp_object: Response object
        :free_position: Free the position? boolean
        """
        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            spectrum_like._background_spectrum,
            bkg_base_array,
            #bkg_parameters,
            free_position,
            spectrum_like._verbose,
        )







# One plugin for Photopeak, one for full response (with Dispersion)
class SPILikeGRB(DispersionSpectrumLike):
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
            **kwargs
    ):
        """
        :param name: name of instance
        :param pyspi_setup: Pyspi Setup object
        """
        self._free_position = free_position

        assert isinstance(
                observation.response, SPIDRM
            ), "The response associated with the observation is not a SPIDRM"

        super(SPILikeGRB, self).__init__(name,
                                         observation,
                                         background,
                                         verbose,
                                         **kwargs)

    def set_model(self, likelihood_model):
        """
        Set the model to be used in the joint minimization.
        :param likelihood_model: likelihood model instance
        """

        super(SPILikeGRB, self).set_model(likelihood_model)

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

                #self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                #    lower_bound=0.0, upper_bound=360
                #)
                #self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                #    lower_bound=-90.0, upper_bound=90
                #)

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector
        """
        TODO need this psd mask stuff later
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
        """
        if precalc_fluxes is None:
            precalc_fluxes = self._integral_flux()

    def get_model(self, precalc_fluxes=None):

        if self._free_position:

            # assumes that the is only one point source which is how it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector

        return super(SPILikeGRB, self).get_model(precalc_fluxes=precalc_fluxes)


    @classmethod
    def from_spectrumlike(
            cls, spectrum_like, free_position=False
    ):
        """
        Generate SPILikeGRB from an existing SpectrumLike child
        :param spectrum_like: SpectrumLike child
        :param rsp_object: Response object
        :free_position: Free the position? boolean
        """
        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            spectrum_like._background_spectrum,
            free_position,
            spectrum_like._verbose,
        )



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


    def _evaluate_model(self, precalc_fluxes=None):
        """
        Evaluate model by multiplying the average effective area of all Ebins with
        the flux in the corresponding incoming Ebin.

        This is photopeak analysis specific! For the general case with energy dispersion, this function needs to
        be changed.

        :param true_fluxes: Flux of incoming spectrum; if None than the flux is computed with the current
        parameters
        """
        if precalc_fluxes is None:
            precalc_fluxes = self._integral_flux()

        if np.any(self._psd_mask):
            self._psd_eff_area[self._psd_mask] = self._like_model.psd_eff_spi.value

        return self._psd_eff_area*self._rsp.effective_area*precalc_fluxes

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

                #self._like_model.point_sources[key].position.ra.prior = Uniform_prior(
                #    lower_bound=0.0, upper_bound=360
                #)
                #self._like_model.point_sources[key].position.dec.prior = Cosine_Prior(
                #    lower_bound=-90.0, upper_bound=90
                #)

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

    def get_model(self, precalc_fluxes=None):

        if self._free_position:

            # assumes that the is only one point source which is how it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._rsp.set_location(ra, dec) # For photopeak not classical "response" matrix but photopeak response vector

        return super(SPILikeGRBPhotopeak, self).get_model(precalc_fluxes=precalc_fluxes)


    @classmethod
    def from_spectrumlike(
            cls, spectrum_like, rsp_object, free_position=False
    ):
        """
        Generate SPILikeGRB from an existing SpectrumLike child
        :param spectrum_like: SpectrumLike child
        :param rsp_object: Response object
        :free_position: Free the position? boolean
        """
        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            spectrum_like._background_spectrum,
            free_position,
            spectrum_like._verbose,
            rsp_object
        )
