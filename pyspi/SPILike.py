import numpy as np
import pandas as pd
import collections

import pyspi
from spi_response import *
from spi_pointing import *
from spi_background import *

from astromodels import Parameter
from threeML.plugin_prototype import PluginPrototype

import scipy.integrate as integrate
import scipy.interpolate as interpolate

__instrument_name = "INTEGRAL-SPI"

class SPILike(PluginPrototype):


    def __init__(self,name,source_name=None,likelihood_model=None):




        # Create the dictionary of nuisance parameters
        # these hold the sky and background fraction
        # for the SPI mask fit.
        # TODO: I have just set them to arbitrary values

        
        self._nuisance_parameters = collections.OrderedDict()

        param_name = "%s_sky" % name

        self._nuisance_parameters[param_name] = Parameter(param_name, 1.0, min_value=0.5, max_value=1.5, delta=0.01)
        self._nuisance_parameters[param_name].fix = True

        param_name = "%s_bkg" % name

        self._nuisance_parameters[param_name] = Parameter(param_name, 1.0, min_value=0.0, max_value=10000.0, delta=0.01)
        self._nuisance_parameters[param_name].fix = False

        # This will contain the JointLikelihood object after a call to .fit()
        self._joint_like_obj = None

        self._likelihood_model = likelihood_model
        
        # This is the name of the source this SED refers to (if it is a SED)
        self._source_name = source_name
        
        #Init SPI response
        self._spi = SPIResponse()
        #Init SPI GRB background
        self._spibg = SPIBackground()
        #Init SPI pointing
        # TODO: most of this here is for testing the general structure
        self._pointing = SPIPointing('data/sc_orbit_param.fits.gz')

        # ONLY FOR TESTING THIS
        data_file = fits.open('data/spi_oper.fits.gz')
        times_sgl     = data_file['SPI.-OSGL-ALL'].data['TIME']
        energies_sgl  = data_file['SPI.-OSGL-ALL'].data['ENERGY']
        detectors_sgl = data_file['SPI.-OSGL-ALL'].data['DETE']
    
        times_psd     = data_file['SPI.-OPSD-ALL'].data['TIME']
        energies_psd  = data_file['SPI.-OPSD-ALL'].data['ENERGY']
        detectors_psd = data_file['SPI.-OPSD-ALL'].data['DETE']
        
        tstart = 4263.11378685185 + 100/86400.
        tstop  = 4263.11413407407 + 100/86400.
        telapse = tstop-tstart
        telapse *= 86400
        
        tedx_sgl = ((times_sgl >= tstart) & (times_sgl <= tstop) & \
                    (energies_sgl >= 20) & (energies_sgl <= 100))
        tedx_psd = ((times_psd >= tstart) & (times_psd <= tstop) & \
                    (energies_psd >= 20) & (energies_psd <= 100))
        
        data_energies_sgl = energies_sgl[tedx_sgl]
        data_energies_psd = energies_psd[tedx_psd]
        self.energies = np.append(data_energies_sgl,data_energies_psd)
        data_detectors_sgl = detectors_sgl[tedx_sgl]
        data_detectors_psd = detectors_psd[tedx_psd]
        self.detectors = np.append(data_detectors_sgl,data_detectors_psd)
        data_evtstypes_sgl = np.zeros(len(np.where(tedx_sgl == True)[0]),dtype=np.int)
        data_evtstypes_psd = np.ones(len(np.where(tedx_psd == True)[0]),dtype=np.int)
        self.evtstypes = np.append(data_evtstypes_sgl,data_evtstypes_psd)

        src_ra  = self._likelihood_model.get_point_source_position(0)[0]
        src_dec = self._likelihood_model.get_point_source_position(0)[1]
                
        zenazi  = self._pointing._calc_zenazi(src_ra,src_dec)
        # choose the index pdx = 4 for testing TBD!
        pdx = 4
        za = np.deg2rad(zenazi[pdx])
        
        self.aeff = self._spi.get_effective_area(za[1],za[0],self.energies,self.detectors)
        # no integration ov er tector bcause model is only in enregy
        # fuck yeah, integration  per derector add in likelihood
        self.aeff_func = []
        for det_num in range(self._spi._n_dets):
            self.aeff_func.append(self._spi.interpolated_effective_area(za[1],za[0])[det_num])

        self.bgm = self._spibg.get_bg_pattern(self.energies,self.detectors,self.evtstypes)

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
        
        expectation_sky = self._likelihood_model.point_sources[self._source_name](self.energies)*self.aeff*telapse
        expectation_bg  = self.bgm*telapse*self._nuisance_parameters[self.name+"_bkg"].value
        
        int_model_bg  = np.sum(self._spibg._bg_data_sgl[:,0:80]+self._spibg._bg_data_psd[:,0:80])*telapse*self._nuisance_parameters[self.name+"_bkg"].value
        int_model_sky = []
        for det_num in range(19):
            int_model_sky.append(integrate.quad(lambda x: self._likelihood_model.point_sources[self._source_name](x)*self.aeff_func[det_num](x)*telapse,20,100)[0])
        
        int_model_tot = int_model_bg + np.sum(int_model_sky)
        
        expectation_tot = expectation_bg + expectation_sky

#        logL = - np.sum(expectation_bg) + np.sum(np.log(expectation_bg))
        logL = - int_model_tot + np.sum(np.log(expectation_tot))

#        logL = integrate.quad(expectation_tot,20,8000)[0] - np.sum(np.log(expectation_tot))

        return logL

    def inner_fit(self):

        # here we fix the model parameters
        # and free the mask parameters. TBD.

        logL = self.get_log_like()

        return logL

    def display(self):

        pass
