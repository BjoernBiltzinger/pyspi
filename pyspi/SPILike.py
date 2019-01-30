import numpy as np
import pandas as pd
import collections

import pyspi
from spi_response import *
from spi_pointing import *


from astromodels import Parameter
from threeML.plugin_prototype import PluginPrototype

__instrument_name = "INTEGRAL-SPI"

class SPILike(PluginPrototype):


    def __init__(self,name,source_name=None):




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

        # This will contain the JointLikelihood object after a call to .fit()
        self._joint_like_obj = None

        self._likelihood_model = None
        
        # This is the name of the source this SED refers to (if it is a SED)
        self._source_name = source_name
        
        #Init SPI response
        self._spi = SPIResponse()
        #Init SPI pointing
        # TODO: most of this here is for testing the general structure
        self._pointing = SPIPointing('data/sc_orbit_param.fits.gz')

        super(SPILike, self).__init__(name, self._nuisance_parameters)

    def set_active_measurements(self,*selections):

        pass

    def display_detector_mask(self):

        pass

        
    def set_model(self, likelihood_model_instance):
        """
        Set the model to be used in the joint minimization. Must be a LikelihoodModel instance.
        """

        self._likelihood_model = likelihood_model_instance
        #pass
    
    def get_log_like(self):
        """
        Return the value of the log-likelihood with the current values for the
        parameters
        """

        telapse = 30.
        
        src_ra  = self._likelihood_model.get_point_source_position(0)[0]
        src_dec = self._likelihood_model.get_point_source_position(0)[1]
                
        zenazi  = self._pointing._calc_zenazi(src_ra,src_dec)
        # choose the index pdx = 4 for testing TBD!
        pdx = 4
        za = np.deg2rad(zenazi[pdx])
        
        # test arrays
        energies  = np.arange(100,500,5)
        detectors = np.zeros(80,dtype=np.int)
        #detectors = np.random.randint(6,high=17,size=80)
        
        aeff = self._spi.get_effective_area(za[1],za[0],energies,detectors)


        expectation = self._likelihood_model.point_sources[self._source_name](energies)*aeff*telapse

        logL = np.sum(np.log(expectation))

        return logL

    def inner_fit(self):

        # here we fix the model parameters
        # and free the mask parameters. TBD.

        logL = self.get_log_like()

        return logL

    def display(self):

        pass
