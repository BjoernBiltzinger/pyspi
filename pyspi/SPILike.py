import collections
from typing import Optional
import numpy as np

from astromodels import Parameter, Model
from astromodels.functions.priors import Cosine_Prior, Uniform_prior

from threeML import PluginPrototype
from threeML.io.file_utils import sanitize_filename
from threeML.plugins.DispersionSpectrumLike import DispersionSpectrumLike
from threeML.plugins.SpectrumLike import SpectrumLike
from threeML.io.logging import setup_logger

from pyspi.utils.response.spi_drm import SPIDRM

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
            verbose: bool = True,
            **kwargs
    ):
        """
        :param name: Name of plugin
        :param observation: observed spectrum
        :param background: background spectrum
        :param bkg_base_array: Base array for background model
        :param free_position: Free the position in the fit?
        :param verbose: Verbose?
        :return: Object
        """
        self._free_position: bool = free_position

        if not isinstance(
                observation.response, SPIDRM
        ):

            log.error("The response associated with the observation"
                      " is not a SPIDRM")

            raise AssertionError()

        super(SPILike, self).__init__(name,
                                      observation,
                                      background,
                                      verbose,
                                      **kwargs)

        self._bkg_base_array = bkg_base_array
        self._bkg_array = np.ones(len(self._bkg_base_array))

    def set_model(self, likelihood_model: Model) -> None:
        """
        Set the model to be used in the joint minimization.
        :param likelihood_model: likelihood model instance
        :return:
        """

        super(SPILike, self).set_model(likelihood_model)

        if self._free_position:
            log.info(f"Freeing the position of {self.name} and setting priors")
            
            for key in self._like_model.point_sources.keys():

                self._like_model.point_sources[key].position.ra.free = True
                self._like_model.point_sources[key].position.dec.free = True

                self._like_model.point_sources[key].position.ra.prior = \
                    Uniform_prior(lower_bound=0.0, upper_bound=360)
                self._like_model.point_sources[key].position.dec.prior = \
                    Cosine_Prior(lower_bound=-90.0, upper_bound=90)

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value
        else:

            for key in self._like_model.point_sources.keys():

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        self._response.set_location(ra, dec)

    def _evaluate_model(self, precalc_fluxes=None):
        """
        Evaluate the model
        :param precalc_fluxes: Precaclulated flux of spectrum
        :return: model counts
        """

        source = super(SPILike, self)._evaluate_model(precalc_fluxes=
                                                      precalc_fluxes)
        self._update_bkg_array()
        bkg = self._bkg_array*self._bkg_base_array
        return source+bkg

    def get_model(self, precalc_fluxes: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get the model
        :param precalc_fluxes: Precaclulated flux of spectrum
        :return: model counts
        """
        if self._free_position:

            # assumes that the is only one point source which is how
            # it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._response.set_location(ra, dec)

        return super(SPILike, self).get_model(precalc_fluxes=precalc_fluxes)

    def _add_bkg_nuisance_parameter(self, bkg_parameters) -> None:
        """
        Add the bkg parameter. Are saved as array.
        :param bkg_parameters:
        :returns:
        """
        
        self._bkg_parameters = bkg_parameters
        for parameter in bkg_parameters:
            self.nuisance_parameters[parameter.name] = parameter
        self._bkg_array = np.ones(len(bkg_parameters))

    def _update_bkg_array(self) -> None:
        """
        Update the array with the background parameter
        :returns:
        """
        
        for key in self._like_model.parameters.keys():
            if "bkg" in key:
                idx = int(key.split("_")[-1])
                self._bkg_array[idx] = self._like_model.parameters[key].value

    def set_free_position(self, flag):
        """
        Set the free position flag
        :param flag: True or False
        :return:
        """
        self._free_position = flag

    @classmethod
    def from_spectrumlike(
        cls,
        spectrum_like,
        bkg_base_array,
        free_position=False
    ):
        """
        Generate SPILikeGRB from an existing SpectrumLike child
        :param spectrum_like: SpectrumLike child
        :param rsp_object: Response object
        :free_position: Free the position? boolean
        :return: Initialized Object
        """
        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            spectrum_like._background_spectrum,
            bkg_base_array,
            free_position,
            spectrum_like._verbose,
        )


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
        :param name: Name of plugin
        :param observation: observed spectrum
        :param background: background spectrum
        :param free_position: Free the position in the fit?
        :param verbose: Verbose?
        :return: Object
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
        :return:
        """

        super(SPILikeGRB, self).set_model(likelihood_model)

        if self._free_position:
            print("Freeing the position of %s and setting priors" % self.name)
            for key in self._like_model.point_sources.keys():
                self._like_model.point_sources[key].position.ra.free = True
                self._like_model.point_sources[key].position.dec.free = True

                self._like_model.point_sources[key].position.ra.prior = \
                    Uniform_prior(lower_bound=0.0, upper_bound=360)
                self._like_model.point_sources[key].position.dec.prior = \
                    Cosine_Prior(lower_bound=-90.0, upper_bound=90)

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value
        else:

            for key in self._like_model.point_sources.keys():

                ra = self._like_model.point_sources[key].position.ra.value
                dec = self._like_model.point_sources[key].position.dec.value

        self._response.set_location(ra, dec)

    def get_model(self, precalc_fluxes=None):
        """
        Get the model
        :param precalc_fluxes: Precaclulated flux of spectrum
        :return: model counts
        """
        if self._free_position:

            # assumes that the is only one point source which is how
            # it should be!
            ra, dec = self._like_model.get_point_source_position(0)

            self._response.set_location(ra, dec)

        return super(SPILikeGRB, self).get_model(precalc_fluxes=precalc_fluxes)

    def set_free_position(self, flag):
        """
        Set the free position flag
        :param flag: True or False
        :return:
        """
        self._free_position = flag

    @classmethod
    def from_spectrumlike(
            cls, spectrum_like, free_position=False
    ):
        """
        Generate SPILikeGRB from an existing SpectrumLike child
        :param spectrum_like: SpectrumLike child
        :param rsp_object: Response object
        :free_position: Free the position? boolean
        :return: Initialized Object
        """
        return cls(
            spectrum_like.name,
            spectrum_like._observed_spectrum,
            spectrum_like._background_spectrum,
            free_position,
            spectrum_like._verbose,
        )
