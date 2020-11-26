from configya import YAMLConfig
from datetime import datetime
import yaml
import time

import numpy as np

default_GRB = {'Unique_analysis_name': None,
               'Special_analysis': 'GRB',
               'Time_of_GRB_UTC' : '101010 010101',
               'Active_Time': '0-100',
               'Background_time_interval_1': '-100--10',
               'Background_time_interval_2': '110-200',
               'Simulate': None,
               'Energy_binned': True,
               'Ebounds': np.logspace(np.log10(20), np.log10(8000), 30).tolist(),
               'Use_only_photopeak': False}


default_CS = {'Unique_analysis_name': None,
               'Special_analysis': 'Constant_Source',
               'Event_types': ['single'],
               'Detectors_to_use': 'All',
               'emin': '20',
               'emax': '8000',
               'Simulate': None,
               'Bkg_estimation': 'Polynominal',
               'Pointings':np.array(['010300010010']).tolist(),
               'Energy_binned': True,
               'Ebounds': np.logspace(np.log10(20), np.log10(8000), 30).tolist(),
               'Use_only_photopeak': False}

           

class GRBConfig(object):
    """
    Class to build the config yaml file used in to specify which analysis should be done.
    """

    def __init__(self, **kwargs):
        """
        Init the config file for a GRB analysis
        """
        self._config = Config(default_GRB)

        for key, value in kwargs.items():
            self._config[key] = value
            
    def change_config(self, **kwargs):
        """
        Change the current config
        :param kwargs: dict with the keywords and new values
        :return:
        """
        for key, value in kwargs.items():
            if key == 'Ebounds':
                value = value.tolist()
            self._config[key] = value
            
    def display(self):
        """
        Display current config file
        :return:
        """
        print(yaml.dump(self._config))

    @property
    def config(self):
        return self._config

class ConstantSourceConfig(object):
    """
    Class to build the config yaml file used in to specify which analysis should be done.
    """

    def __init__(self, **kwargs):
        """
        Init the config file for a GRB analysis
        """
        self._config = Config(default_CS)
        for key, value in kwargs.items():
            self._config[key] = value
            
    def change_config(self, **kwargs):
        """
        Change the current config
        :param kwargs: dict with the keywords and new values
        :return:
        """
        for key, value in kwargs.items():
            if key=='Ebounds' or key=='Pointings':
                value=value.tolist()
            self._config[key] = value
            
    def display(self):
        """
        Display current config file
        :return:
        """
        print(yaml.dump(self._config))

    @property
    def config(self):
        return self._config

class Config(YAMLConfig):
    
    def __init__(self, default_config):
        
        time_stamp = datetime.now().strftime("%d%m%y_%H%M%S%f")
        time.sleep(0.1)
        super(Config, self).__init__(default_config,
                                       '~/.pyspi',
                                       f'config_{time_stamp}.yml')
