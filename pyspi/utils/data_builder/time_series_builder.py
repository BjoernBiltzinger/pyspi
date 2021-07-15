from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder

from astropy.time.core import Time, TimeDelta
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion, BinnedSpectrum

import yaml
import os
import numpy as np
from datetime import datetime
import h5py
from astropy.io import fits


from pyspi.io.get_files import get_files_afs, get_files_isdcarc
from pyspi.io.package_data import get_path_of_external_data_dir, \
    get_path_of_data_file
from pyspi.utils.detector_ids import double_names, triple_names
from pyspi.config.config_builder import Config
from pyspi.utils.function_utils import construct_energy_bins, find_needed_ids, \
    ISDC_MJD_to_cxcsec, leapseconds
from threeML.io.file_utils import sanitize_filename
from threeML.utils.time_series.event_list import EventListWithDeadTime
from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
class SPISWFile(object):
    def __init__(self, det, pointing_id, ebounds):
        """
        Class to read in all the data needed from a SCW file for a given config file
        :param config: Config yml filename, Config object or dict
        :param det: For which detector?
        """
        # General nameing
        self._det_name = f"Detector {det}"
        self._mission = "Integral/SPI"
        self._ebounds = ebounds
        # How many echans?
        self._n_channels = len(self._ebounds)-1

        # Check that det is a valid number
        self._det = det
        if self._det != "singles":
            assert self._det in np.arange(85), f"{self._det} is not a valid detector. Please only use detector ids between 0 and 84."

        # Find the SW ID of the data file we need for this time
        self._pointing_id = pointing_id

        # Get the data, either from afs or from ISDC archive
        try:
            # Get the data from the afs server
            get_files_afs(self._pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(self._pointing_id)

        # Read in all we need
        self._read_in_pointing_data(self._pointing_id)

    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id
        :param pointing_id: pointing_id for which we want the data
        :return:
        """

        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:

            # Get time of first and last event (t0 at grb time)
            time_sgl = ISDC_MJD_to_cxcsec(hdu_oper[1].data['time'])
            time_psd = ISDC_MJD_to_cxcsec(hdu_oper[2].data['time'])
            time_me2 = ISDC_MJD_to_cxcsec(hdu_oper[4].data['time'])
            time_me3 = ISDC_MJD_to_cxcsec(hdu_oper[5].data['time'])

            self._time_start = np.min(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))
            self._time_stop = np.max(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))

            # Read in the data for the wanted detector
            # For single events we have to take both the non_psd (often called sgl here...)
            # and the psd events. Both added together give the real single events.
            if self._det in range(19) or self._det=="singles":
                dets_sgl = hdu_oper[1].data['DETE']
                if self._det != "singles":
                    time_sgl = time_sgl[dets_sgl == self._det]
                    energy_sgl = hdu_oper[1].data['energy'][dets_sgl == self._det]
                else:
                    energy_sgl = hdu_oper[1].data['energy']
                #if "psd" in self._event_types:
                #if self._use_psd:
                dets_psd = hdu_oper[2].data['DETE']
                if self._det != "singles":
                    time_psd = time_psd[dets_psd == self._det]
                    energy_psd = hdu_oper[2].data['energy'][dets_psd == self._det]
                else:
                    energy_psd = hdu_oper[2].data['energy']

            if self._det in range(19, 61):
                dets_me2 = np.sort(hdu_oper[4].data['DETE'], axis=1)
                i, k = double_names[self._det]
                mask = np.logical_and(dets_me2[:, 0] == i,
                                      dets_me2[:, 1] == k)

                time_me2 = time_me2[mask]
                energy_me2 = np.sum(hdu_oper[4].data['energy'][mask], axis=1)

            if self._det in range(61,85):
                dets_me3 = np.sort(hdu_oper[5].data['DETE'], axis=1)
                i, j, k = triple_names[self._det]
                mask = np.logical_and(np.logical_and(dets_me3[:, 0] == i,
                                                     dets_me3[:, 1] == j),
                                      dets_me3[:, 2] == k)

                time_me3 = time_me3[mask]
                energy_me3 = np.sum(hdu_oper[5].data['energy'][mask], axis=1)

        if self._det in range(19) or self._det=="singles":

            self._times = time_psd
            self._energies = energy_psd

            # Don't add the non-psd single events in the electronic noise range
            # We will account for this later by a extra parameter determining
            # the fraction of psd events in this energy range


            ## turn this of for the moment########

            #self._times = np.append(self._times, time_sgl[~np.logical_and(energy_sgl > 1400,
            #                                                              energy_sgl < 1700)])
            #self._energies = np.append(self._energies, energy_sgl[~np.logical_and(energy_sgl > 1400,
            #                                                                      energy_sgl < 1700)])
            ################
            self._times = np.append(self._times, time_sgl)
            self._energies = np.append(self._energies, energy_sgl)
            # sort in time
            sort_array = np.argsort(self._times)
            self._times = self._times[sort_array]
            self._energies = self._energies[sort_array]

        if self._det in range(19, 61):

            self._times = time_me2
            self._energies = energy_me2

        if self._det in range(61, 85):

            self._times = time_me3
            self._energies = energy_me3

        # Check if there are any counts
        if np.sum(self._energies) == 0:

            raise AssertionError(f"The detector {self._det} has zero counts and is therefore not active."\
                                 "Please exclude this detector!")

        # Bin this in the energy bins we have
        self._energy_bins = np.ones_like(self._energies, dtype=int)*-1
        # Loop over ebins
        for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
            mask = np.logical_and(self._energies>emin, self._energies<emax)
            self._energy_bins[mask] = np.ones_like(self._energy_bins[mask])*i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        mask = self._energy_bins == -1
        self._energy_bins = self._energy_bins[~mask]
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]

    @property
    def geometry_file_path(self):
        """
        Path to the spacecraft geometry file
        """
        return os.path.join(get_path_of_external_data_dir(), 'pointing_data', self._pointing_id, 'sc_orbit_param.fits.gz')

    @property
    def times(self):
        return self._times

    @property
    def energies(self):
        return self._energies

    @property
    def energy_bins(self):
        return self._energy_bins

    @property
    def ebounds(self):
        return self._ebounds

    @property
    def det(self):
        return self._det

class SPISWFileGRB(object):

    def __init__(self, config, det):
        """
        Class to read in all the data needed from a SCW file for a given config file
        :param config: Config yml filename, Config object or dict
        :param det: For which detector?
        """

        # Read in config file
        # Check if config is a dict
        if not isinstance(config, dict):

            # If not, check if it is a Config object (from Config_builder)
            if isinstance(config, Config):
                configuration = config
            else:
                # If not assume this is a file name
                configuration_file = sanitize_filename(config)

                assert os.path.exists(config), "Configuration file %s does not exist" % configuration_file

                # Read the configuration
                with open(configuration_file) as f:

                    configuration = yaml.safe_load(f)

        else:

            # Configuration is a dictionary. Nothing to do
            configuration = config


        # General nameing
        self._det_name = f"Detector {det}"
        self._mission = "Integral/SPI"

        # Binned or unbinned analysis?
        self._binned = configuration['Energy_binned']
        if self._binned:
            # Set ebounds of energy bins
            self._ebounds = np.array(configuration['Ebounds'])

            # If no ebounds are given use the default ones
            assert self._ebounds is not None, "Please give the bounds for the Ebins."

            # Construct final energy bins (make sure to make extra echans for the electronic noise energy range)
            self._ebounds, _  = construct_energy_bins(self._ebounds)
        else:
            raise NotImplementedError('Unbinned analysis not implemented!')

        # How many echans?
        self._n_channels = len(self._ebounds)-1

        # Time bounds for bkg polynominal and active time
        self._bkg_time_1 = configuration['Background_time_interval_1']
        self._bkg_time_2 = configuration['Background_time_interval_2']
        self._active_time = configuration['Active_Time']

        # Time of GRB. Needed to get the correct pointing.
        time_of_grb = configuration['Time_of_GRB_UTC']
        time = datetime.strptime(time_of_grb, '%y%m%d %H%M%S')
        self._time_of_GRB = Time(time)

        # Check that det is a valid number
        self._det = det
        assert self._det in np.arange(85), f"{self._det} is not a valid detector. Please only use detector ids between 0 and 84."

        # Find the SW ID of the data file we need for this time
        self._pointing_id = find_needed_ids(self._time_of_GRB)

        # Get the data, either from afs or from ISDC archive
        try:
            # Get the data from the afs server
            get_files_afs(self._pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(self._pointing_id)

        # Read in all we need
        self._read_in_pointing_data(self._pointing_id)

    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id
        :param pointing_id: pointing_id for which we want the data
        :return:
        """
        # Reference time of GRB
        GRB_ref_time_cxcsec = ISDC_MJD_to_cxcsec((self._time_of_GRB+leapseconds(self._time_of_GRB)).tt.mjd-51544)

        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:

            # Get time of first and last event (t0 at grb time)
            time_sgl = ISDC_MJD_to_cxcsec(hdu_oper[1].data['time']) - GRB_ref_time_cxcsec
            time_psd = ISDC_MJD_to_cxcsec(hdu_oper[2].data['time']) - GRB_ref_time_cxcsec
            time_me2 = ISDC_MJD_to_cxcsec(hdu_oper[4].data['time']) - GRB_ref_time_cxcsec
            time_me3 = ISDC_MJD_to_cxcsec(hdu_oper[5].data['time']) - GRB_ref_time_cxcsec

            self._time_start = np.min(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))
            self._time_stop = np.max(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))

            # Read in the data for the wanted detector
            # For single events we have to take both the non_psd (often called sgl here...)
            # and the psd events. Both added together give the real single events.
            if self._det in range(19):
                dets_sgl = hdu_oper[1].data['DETE']
                time_sgl = time_sgl[dets_sgl == self._det]
                energy_sgl = hdu_oper[1].data['energy'][dets_sgl == self._det]

                #if "psd" in self._event_types:
                #if self._use_psd:
                dets_psd = hdu_oper[2].data['DETE']
                time_psd = time_psd[dets_psd == self._det]
                energy_psd = hdu_oper[2].data['energy'][dets_psd == self._det]

            if self._det in range(19, 61):
                dets_me2 = np.sort(hdu_oper[4].data['DETE'], axis=1)
                i, k = double_names[self._det]
                mask = np.logical_and(dets_me2[:, 0] == i,
                                      dets_me2[:, 1] == k)

                time_me2 = time_me2[mask]
                energy_me2 = np.sum(hdu_oper[4].data['energy'][mask], axis=1)

            if self._det in range(61,85):
                dets_me3 = np.sort(hdu_oper[5].data['DETE'], axis=1)
                i, j, k = triple_names[self._det]
                mask = np.logical_and(np.logical_and(dets_me3[:, 0] == i,
                                                     dets_me3[:, 1] == j),
                                      dets_me3[:, 2] == k)

                time_me3 = time_me3[mask]
                energy_me3 = np.sum(hdu_oper[5].data['energy'][mask], axis=1)

        if self._det in range(19):

            self._times = time_psd
            self._energies = energy_psd

            # Don't add the non-psd single events in the electronic noise range
            # We will account for this later by a extra parameter determining
            # the fraction of psd events in this energy range


            ## turn this of for the moment########

            self._times = np.append(self._times, time_sgl[~np.logical_and(energy_sgl > 1400,
                                                                          energy_sgl < 1700)])
            self._energies = np.append(self._energies, energy_sgl[~np.logical_and(energy_sgl > 1400,
                                                                                  energy_sgl < 1700)])
            ################
            self._times = np.append(self._times, time_sgl)
            self._energies = np.append(self._energies, energy_sgl)
            # sort in time
            sort_array = np.argsort(self._times)
            self._times = self._times[sort_array]
            self._energies = self._energies[sort_array]

        if self._det in range(19, 61):

            self._times = time_me2
            self._energies = energy_me2

        if self._det in range(61, 85):

            self._times = time_me3
            self._energies = energy_me3

        # Check if there are any counts
        if np.sum(self._energies) == 0:

            raise AssertionError(f"The detector {self._det} has zero counts and is therefore not active."\
                                 "Please exclude this detector!")

        # Bin this in the energy bins we have
        self._energy_bins = np.ones_like(self._energies, dtype=int)*-1
        # Loop over ebins
        for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
            mask = np.logical_and(self._energies>emin, self._energies<emax)
            self._energy_bins[mask] = np.ones_like(self._energy_bins[mask])*i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        mask = self._energy_bins == -1
        self._energy_bins = self._energy_bins[~mask]
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]

    @property
    def geometry_file_path(self):
        """
        Path to the spacecraft geometry file
        """
        return os.path.join(get_path_of_external_data_dir(), 'pointing_data', self._pointing_id, 'sc_orbit_param.fits.gz')

    @property
    def times(self):
        return self._times

    @property
    def energies(self):
        return self._energies

    @property
    def energy_bins(self):
        return self._energy_bins

    @property
    def ebounds(self):
        return self._ebounds

    @property
    def det(self):
        return self._det

class TimeSeriesBuilderSPI(TimeSeriesBuilder):

    def __init__(
        self,
        name,
        time_series,
        response=None,
        poly_order=-1,
        unbinned=True,
        verbose=True,
        restore_poly_fit=None,
        container_type=BinnedSpectrumWithDispersion,
        **kwargs
    ):
        """
        Class to build the time_series for SPI. Inherited from the 3ML TimeSeriesBuilder with added
        class methods to build the object for given pyspi config files.
        """

        super(TimeSeriesBuilderSPI, self).__init__(
            name,
            time_series,
            response=response,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_poly_fit,
            container_type=container_type,
            **kwargs
        )

    @classmethod
    def from_spi_grb_config_rmf(cls,
        config,
        det,
        restore_background=None,
        poly_order=0,
        unbinned=True,
        verbose=True,
        response=None
    ):
        """
        Class method to build the time_series_builder from a pyspi grb conifg file
        :param config: Config yml filename, Config object or dict
        :param det: Which det?
        :param restore_background: File to restore bkg
        :param poly_order: Which poly_order? -1 gives automatic determination
        :param unbinned:
        :param verbose:
        """

        spi_grb_setup = SPISWFileGRB(config, det)


        # TODO later with deadtime - at the moment dummy array with 0 dead time for all events
        event_list = EventListWithDeadTime(
            arrival_times=spi_grb_setup.times,
            measurement=spi_grb_setup.energy_bins,
            n_channels=spi_grb_setup._n_channels,
            start_time=spi_grb_setup._time_start,
            stop_time=spi_grb_setup._time_stop,
            dead_time=np.zeros_like(spi_grb_setup.times),
            first_channel=0,
            instrument=spi_grb_setup._det_name,
            mission=spi_grb_setup._mission,
            verbose=verbose,
            edges=spi_grb_setup.ebounds,
        )

        # This build a time_series_object for a photopeak only response with no Dispersion
        # For a real response with dispersion one need to use the BinnedSpectrumWithDispersion
        # Container and input a threeML response object.

        return cls(
            f"IntegralSPIDet{spi_grb_setup.det}",
            event_list,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            response=response,
            container_type=BinnedSpectrumWithDispersion
        )

        
    @classmethod
    def from_spi_grb_config(cls,
        config,
        det,
        restore_background=None,
        poly_order=0,
        unbinned=True,
        verbose=True,
    ):
        """
        Class method to build the time_series_builder from a pyspi grb conifg file
        :param config: Config yml filename, Config object or dict
        :param det: Which det?
        :param restore_background: File to restore bkg
        :param poly_order: Which poly_order? -1 gives automatic determination
        :param unbinned: 
        :param verbose:
        """

        spi_grb_setup = SPISWFileGRB(config, det)


        # TODO later with deadtime - at the moment dummy array with 0 dead time for all events
        event_list = EventListWithDeadTime(
            arrival_times=spi_grb_setup.times,
            measurement=spi_grb_setup.energy_bins,
            n_channels=spi_grb_setup._n_channels,
            start_time=spi_grb_setup._time_start,
            stop_time=spi_grb_setup._time_stop,
            dead_time=np.zeros_like(spi_grb_setup.times),
            first_channel=0,
            instrument=spi_grb_setup._det_name,
            mission=spi_grb_setup._mission,
            verbose=verbose,
            edges=spi_grb_setup.ebounds,
        )

        # This build a time_series_object for a photopeak only response with no Dispersion
        # For a real response with dispersion one need to use the BinnedSpectrumWithDispersion
        # Container and input a threeML response object.

        return cls(
            f"IntegralSPIDet{spi_grb_setup.det}",
            event_list,
            response=None,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedSpectrum
        )

    @classmethod
    def from_spi_grb_sim(cls,
                         sim_object,
                         det,
                         restore_background=None,
                         poly_order=0,
                         unbinned=False,
                         verbose=True):

        event_list = BinnedSpectrumSeries(
            sim_object.get_binned_spectrum_set(det),
            first_channel=0,
            mission="Integral",
            instrument=f"SPIDET{det}",
            verbose=verbose,
        )

        return cls(
            f"IntegralSPIDet{det}",
            event_list,
            response=None,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            container_type=BinnedSpectrum
        )

    @classmethod
    def from_spi_grb_sim_rmf(cls,
                         sim_object,
                         det,
                         restore_background=None,
                         poly_order=0,
                         unbinned=False,
                             verbose=True,
                             response=None):

        event_list = BinnedSpectrumSeries(
            sim_object.get_binned_spectrum_set(det),
            first_channel=0,
            mission="Integral",
            instrument=f"SPIDET{det}",
            verbose=verbose,
        )

        return cls(
            f"IntegralSPIDet{det}",
            event_list,
            poly_order=poly_order,
            unbinned=unbinned,
            verbose=verbose,
            restore_poly_fit=restore_background,
            response=response,
            container_type=BinnedSpectrumWithDispersion
        )

    @classmethod
    def from_spi_constant_pointing(cls,
                                   det,
                                   ebounds,
                                   pointing_id,
                                   response
    ):
        spi_grb_setup1 = SPISWFile(det, pointing_id, ebounds)

        e = EventListWithDeadTime(arrival_times=spi_grb_setup1.times,
                                  measurement=spi_grb_setup1.energy_bins,
                                  n_channels=spi_grb_setup1._n_channels,
                                  start_time=spi_grb_setup1._time_start,
                                  stop_time=spi_grb_setup1._time_stop,
                                  dead_time=np.zeros_like(spi_grb_setup1.times),
                                  first_channel=0,
                                  instrument=spi_grb_setup1._det_name,
                                  mission=spi_grb_setup1._mission,
                                  verbose=False,
                                  edges=spi_grb_setup1.ebounds,
        )

        tsb = cls(f"spi_{pointing_id}_{det}",
                  e,
                  verbose=True,
                  response=response,
                  container_type=BinnedSpectrumWithDispersion)

        tsb.set_active_time_interval(f"{spi_grb_setup1.times[0]}-{spi_grb_setup1.times[-1]}")

        return tsb
