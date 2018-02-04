import numpy as np

from astromodels import Parameter
from threeML.plugin_prototype import PluginPrototype

__instrument_name = "INTEGRAL-SPI"

class SPILike(PluginPrototype):


    def __init__(self,name):




        # Create the dictionary of nuisance parameters
        # these hold the sky and background fraction
        # for the SPI mask fit.
        # TODO: I have just set them to arbitrary values

        
        self._nuisance_parameters = collections.OrderedDict()

        param_name = "%s_sky" % name

        self._nuisance_parameters[param_name] = Parameter(param_name, 1.0, min_value=0.5, max_value=1.5, delta=0.01)
        self._nuisance_parameters[param_name].fix = True

        param_name = "%s_bkg" % name

        self._nuisance_parameters[param_name] = Parameter(param_name, 1.0, min_value=0.5, max_value=1.5, delta=0.01)
        self._nuisance_parameters[param_name].fix = True



        super(SPILike, self).__init__(name, self._nuisance_parameters)

    def set_active_measurements(self,*selections):

        pass

    def display_detector_mask(self):

        pass

        
    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        pass
    
    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

    

        logL = 0.

        return logL

    def inner_fit(self):

        # here we fix the model parameters
        # and free the mask parameters. TBD.

        logL = self.get_log_like()

        return logL

    def display(self):

        pass
