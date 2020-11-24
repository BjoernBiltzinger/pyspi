from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder
from pyspi.spi_response_new import ResponsePhotopeak, ResponseRMF, multi_response_irf_read_objects
from pyspi.spi_pointing import _construct_sc_matrix, _transform_icrs_to_spi, SPIPointing
from astropy.time.core import Time, TimeDelta
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion, BinnedSpectrum
from threeML.utils.time_series.binned_spectrum_series import BinnedSpectrumSeries
import os
import numpy as np
from datetime import datetime
import h5py
from astropy.io import fits


from pyspi.io.get_files import get_files_afs, get_files_isdcarc
from pyspi.io.package_data import get_path_of_external_data_dir, get_path_of_data_file
from pyspi.utils.detector_ids import double_names, triple_names
from pyspi.Config_Builder import Config
from threeML.io.file_utils import sanitize_filename
from threeML.utils.time_series.event_list import EventListWithDeadTime

class SPISWFile(object):

    def __init__(self, config, det):

        if not isinstance(config, dict):

            if isinstance(config, Config):
                configuration = config
            else:
                # Assume this is a file name
                configuration_file = sanitize_filename(config)

                assert os.path.exists(config), "Configuration file %s does not exist" % configuration_file

                # Read the configuration
                with open(configuration_file) as f:

                    configuration = yaml.safe_load(f)

        else:

            # Configuration is a dictionary. Nothing to do
            configuration = config


        self._det_name = f"Detector {det}"
        self._mission = "Integral/SPI"

        # Which energy range?
        self._emin = float(configuration['emin'])
        self._emax = float(configuration['emax'])

        # Binned or unbinned analysis?
        self._binned = configuration['Energy_binned']
        if self._binned:
            # Set ebounds of energy bins
            self._ebounds = np.array(configuration['Ebounds'])
            # If no ebounds are given use the default ones
            if self._ebounds is None:
                self._ebounds = np.logspace(np.log10(self._emin), np.log10(self._emax), 30)
            # Construct final energy bins (make sure to make extra echans for the electronic noise energy range)
            self._construct_energy_bins()
        else:
            raise NotImplementedError('Unbinned analysis not implemented!')
        self._n_channels = len(self._ebounds)-1
        self._bkg_time_1 = configuration['Background_time_interval_1']
        self._bkg_time_2 = configuration['Background_time_interval_2']

        # Time of GRB. Needed to get the correct pointing.
        time_of_grb = configuration['Time_of_GRB_UTC']
        time = datetime.strptime(time_of_grb, '%y%m%d %H%M%S')
        self._time_of_GRB = Time(time)

        # Active_Time of GRB 'start-stop' format
        self._active_time = configuration['Active_Time']
        self._det = det
        assert self._det in np.arange(85), f"{self._det} is not a valid detector. Please only use detector ids between 0 and 84."

        self._pointing_id = self._find_needed_ids()

        try:
            # Get the data from the afs server
            get_files_afs(self._pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(self._pointing_id)

        self._read_in_pointing_data(self._pointing_id)

    def _construct_energy_bins(self):
        """
        Function to construct the final energy bins that will be used in the analysis.
        Basically only does one thing: If the single events are included in the analysis
        it ensures that no energybin is covering simultaneously energy outside and inside
        of [psd_low_energy, psd_high_energy]. In this area the single detection photons
        that were not tested by the PSD suffer the "electronical noise" and are very unrealiable.
        The events that have passed the PSD test do not suffer from this "electronical noise".
        Thus we want to only use the PSD events in this energy range. Therefore we construct the
        ebins in such a way that there are ebins outside of this energy range, for which we will
        use normal single + psd events and ebins inside of this energy range for which we only
        want to use the psd events.
        :return:
        """

        psd_low_energy = 1400
        psd_high_energy = 1700

        change = False
        # Case 1400-1700 is completly in the ebound range
        if self._ebounds[0]<psd_low_energy and self._ebounds[-1]>psd_high_energy:
            psd_bin = True
            start_found = False
            stop_found = False
            for i, e in enumerate(self._ebounds):
                if e>=psd_low_energy and not start_found:
                    start_index = i
                    start_found=True
                if e>=1700 and not stop_found:
                    stop_index = i
                    stop_found=True
            self._ebounds = np.insert(self._ebounds, start_index, psd_low_energy)
            self._ebounds = np.insert(self._ebounds, stop_index+1, psd_high_energy)

            if stop_index-start_index>1:
                sgl_mask = np.logical_and(np.logical_or(self._ebounds[:-1]<=psd_low_energy,
                                                        self._ebounds[:-1]>=psd_high_energy),
                                          np.logical_or(self._ebounds[1:]<=psd_low_energy,
                                                        self._ebounds[1:]>=psd_high_energy))
            elif stop_index-start_index==1:
                sgl_mask = np.ones(len(self._ebounds)-1, dtype=bool)
                sgl_mask[start_index] = False
                sgl_mask[stop_index] = False
            elif stop_index-start_index==0:
                sgl_mask = np.ones(len(self._ebounds)-1, dtype=bool)
                sgl_mask[start_index] = False
            change = True
        # Upper bound of erange in psd bin
        elif self._ebounds[0]<psd_low_energy and self._ebounds[-1]>psd_low_energy:
            psd_bin = True
            start_found = False
            for i, e in enumerate(a):
                if e>=psd_low_energy and not start_found:
                    start_index = i
                    start_found=True
            self._ebounds = np.insert(self._ebounds, start_index, psd_low_energy)
            sgl_mask = (self._ebounds<psd_low_energy)[:-1]
            change = True
        # Lower bound of erange in psd bin
        elif self._ebounds[0]<psd_high_energy and self._ebounds[-1]>psd_high_energy:
            psd_bin = True
            stop_found = False
            for i, e in enumerate(a):
                if e>=psd_high_energy and not stop_found:
                    stop_index = i
                    stop_found=True
            self._ebounds = np.insert(self._ebounds, stop_index, psd_high_energy)
            sgl_mask = (self._ebounds>=psd_high_energy)[:-1]
            change=True
        # else erange completly outside of psd bin => all just single
        else:
            sgl_mask = np.ones_like(self._ebounds[:-1], dtype=bool)

        self._sgl_mask = sgl_mask

        if change:
            self._use_ele_noise = True
            print('I had to readjust the ebins to avoid having ebins inside of the single event electronic noise energy range. The new boundaries of the ebins are: {}.'.format(self._ebounds))
        else:
            self._use_ele_noise = False

    def _find_needed_ids(self):
        """
        Get the pointing id of the needed data to cover the GRB time
        :return: Needed pointing id
        """

        # Path to file, which contains id information and start and stop
        # time
        id_file_path = get_path_of_data_file('id_data_time.hdf5')

        # Get GRB time in ISDC_MJD
        self._time_of_GRB_ISDC_MJD = self._ISDC_MJD(self._time_of_GRB)

        # Get which id contain the needed time. When the wanted time is
        # too close to the boundarie also add the pervious or following
        # observation id
        id_file = h5py.File(id_file_path, 'r')
        start_id = id_file['Start'].value
        stop_id = id_file['Stop'].value
        ids = id_file['ID'].value

        mask_larger = start_id < self._time_of_GRB_ISDC_MJD
        mask_smaller = stop_id > self._time_of_GRB_ISDC_MJD

        try:
            id_number = list(mask_smaller*mask_larger).index(True)
            print('Needed data is stored in pointing_id: {}'.format(ids[id_number]))
        except:
            raise Exception('No pointing id contains this time...')

        return ids[id_number].decode("utf-8")

    def _ISDC_MJD(self, time_object):
        """
        :param time_object: Astropy time object of grb time
        :return: Time in Integral MJD time
        """

        #TODO Check time. In integral file ISDC_MJD time != DateStart. Which is correct?
        return (time_object+self._leapseconds(time_object)).tt.mjd-51544

    def _leapseconds(self, time_object):
        """
        Hard coded leap seconds from start of INTEGRAL to time of time_object
        :param time_object: Time object to which the number of leapseconds should be detemined
        :return: TimeDelta object of the needed leap seconds
        """
        if time_object<Time(datetime.strptime('060101 000000', '%y%m%d %H%M%S')):
            lsec = 0
        elif time_object<Time(datetime.strptime('090101 000000', '%y%m%d %H%M%S')):
            lsec = 1
        elif time_object<Time(datetime.strptime('120701 000000', '%y%m%d %H%M%S')):
            lsec = 2
        elif time_object<Time(datetime.strptime('150701 000000', '%y%m%d %H%M%S')):
            lsec = 3
        elif time_object<Time(datetime.strptime('170101 000000', '%y%m%d %H%M%S')):
            lsec = 4
        else:
            lsec=5
        return TimeDelta(lsec, format='sec')

    def _ISDC_MJD_to_cxcsec(self, ISDC_MJD_time):
        """
        Convert ISDC_MJD to UTC
        :param ISDC_MJD_time: time in ISDC_MJD time format
        :return: time in cxcsec format (seconds since 1998-01-01 00:00:00)
        """
        return Time(ISDC_MJD_time+51544, format='mjd', scale='utc').cxcsec

    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given pointing_id
        :param pointing_id: pointing_id for which we want the data
        :return:
        """
        # Reference time of GRB
        GRB_ref_time_cxcsec = self._ISDC_MJD_to_cxcsec(self._time_of_GRB_ISDC_MJD)

        with fits.open(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')) as hdu_oper:

            # Get time of first and last event (t0 at grb time)
            time_sgl = self._ISDC_MJD_to_cxcsec(hdu_oper[1].data['time']) - GRB_ref_time_cxcsec
            time_psd = self._ISDC_MJD_to_cxcsec(hdu_oper[2].data['time']) - GRB_ref_time_cxcsec
            time_me2 = self._ISDC_MJD_to_cxcsec(hdu_oper[4].data['time']) - GRB_ref_time_cxcsec
            time_me3 = self._ISDC_MJD_to_cxcsec(hdu_oper[5].data['time']) - GRB_ref_time_cxcsec

            self._time_start = np.min(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))
            self._time_stop = np.max(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))

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
                i, k = double_names[self._det-19]
                mask = np.logical_and(dets_me2[:, 0] == i,
                                      dets_me2[:, 1] == k)

                time_me2 = time_me2[mask]
                energy_me2 = np.sum(hdu_oper[4].data['energy'][mask], axis=1)

            if self._det in range(61,85):
                dets_me3 = np.sort(hdu_oper[5].data['DETE'], axis=1)
                i, j, k = tripple_names[self._det-61]
                mask = np.logical_and(np.logical_and(dets_me3[:, 0] == i,
                                                     dets_me3[:, 1] == j),
                                      dets_me3[:, 2] == k)

                time_me3 = time_me3[mask]
                energy_me3 = np.sum(hdu_oper[5].data['energy'][mask], axis=1)

        if self._det in range(19):

            self._times = time_psd
            self._energies = energy_psd

            #if self._use_ele_noise:
            # Don't add the non-psd single events in the electronic noise range
            self._times = np.append(self._times, time_sgl[~np.logical_and(energy_sgl>1400,
                                                                          energy_sgl<1700)])
            self._energies = np.append(self._energies, energy_sgl[~np.logical_and(energy_sgl>1400,
                                                                                  energy_sgl<1700)])

            # sort in time
            sort_array = np.argsort(self._times)
            self._times = self._times[sort_array]
            self._energies = self._energies[sort_array]

        if self._det in range(19, 61):

            self._times = time_me2
            self._energies = energy_me2

        if self._det in range(61,85):

            self._times = time_me3
            self._energies = energy_me3

        if np.sum(self._energies)==0:

            raise AssertionError(f"The detector {self._det} has zero counts and is therefore not active."\
                                 "Please exclude this detector!")

        # Bin this in the energy bins we have
        self._energy_bins = np.ones_like(self._energies, dtype=int)*-1
        for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
            mask = np.logical_and(self._energies>emin, self._energies<emax)
            self._energy_bins[mask] = np.ones_like(self._energy_bins[mask])*i

        #Throw away all events that had energies outside of the ebounds that should be used
        mask = self._energy_bins ==-1
        self._energy_bins = self._energy_bins[~mask]
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]

    @property
    def geometry_file_path(self):
        return os.path.join(get_path_of_external_data_dir(), 'pointing_data', self._pointing_id, 'sc_orbit_param.fits.gz')

    @property
    def rsp(self):
        return self._response_object

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
    def from_spi_grb_config(cls,
        config,
        det,
        restore_background=None,
        poly_order=0,
        unbinned=True,
        verbose=True,
    ):

        spi_grb_setup = SPISWFile(config, det)


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
