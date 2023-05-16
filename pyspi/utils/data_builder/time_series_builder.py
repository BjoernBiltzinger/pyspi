from astropy.time.core import Time, TimeDelta
import yaml
import os
import numpy as np
from datetime import datetime
import h5py
from astropy.io import fits
import copy

from pyspi.io.get_files import get_files
from pyspi.io.package_data import get_path_of_external_data_dir
from pyspi.utils.livedets import double_names, triple_names, get_live_dets
from pyspi.utils.function_utils import find_needed_ids, ISDC_MJD_to_cxcsec, \
    get_time_object


from threeML.io.file_utils import sanitize_filename
from threeML.utils.time_series.event_list import EventListWithDeadTime,\
    EventListWithLiveTime
from threeML.utils.time_series.binned_spectrum_series import \
    BinnedSpectrumSeries
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrumWithDispersion
from threeML.utils.data_builders.time_series_builder import TimeSeriesBuilder


class SPISWFile(object):
    def __init__(self, det, pointing_id, ebounds):
        """
        Class to read in all the data needed from a SCW file
        for a given config file

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
        if not isinstance(det, list):
            self._det = [det]
        else:
            self._det = det
        for d in self._det:
            assert d in np.arange(85), f"{d} is not a valid"\
                " detector. Please only use detector ids between 0 and 84."

        # Find the SW ID of the data file we need for this time
        self._pointing_id = pointing_id

        # Get the data, either from afs or from ISDC archive
        try:
            # Get the data from the afs server
            get_files(pointing_id, access="afs")
        except AssertionError:
            # Get the files from the iSDC data archive
            print("AFS data access did not work."
                  " I will try the ISDC data archive.")
            get_files(pointing_id, access="isdc")

        # Read in all we need
        self._read_in_pointing_data(self._pointing_id)
        self._get_deadtime_info()

    def _read_in_pointing_data(self, pointing_id):
        """
        Gets all needed information from the data file for the given
        pointing_id

        :param pointing_id: pointing_id for which we want the data

        :returns:
        """

        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data', pointing_id,
                                    'spi_oper.fits.gz')) as hdu_oper:

            # Get time of first and last event (t0 at grb time)
            time_sgl = ISDC_MJD_to_cxcsec(hdu_oper[1].data['time'])
            time_psd = ISDC_MJD_to_cxcsec(hdu_oper[2].data['time'])
            time_me2 = ISDC_MJD_to_cxcsec(hdu_oper[4].data['time'])
            time_me3 = ISDC_MJD_to_cxcsec(hdu_oper[5].data['time'])

            time_start = np.min(np.concatenate([time_sgl, time_psd,
                                                      time_me2, time_me3]))
            time_stop = np.max(np.concatenate([time_sgl, time_psd,
                                                     time_me2, time_me3]))

            # renorm time start to 0
            time_sgl -= time_start
            time_psd -= time_start
            time_me2 -= time_start
            time_me3 -= time_start
            self._time_start = 0
            self._time_stop = time_stop-time_start

            # save the norm time, we need this when we read in the dead time info
            self._norm_time = time_start

            # Read in the data for the wanted detector
            # For single events we have to take both the non_psd
            # (often called sgl here...)
            # and the psd events. Both added together
            # give the real single events.
            if self._det in range(19) or self._det == "singles":
                dets_sgl = hdu_oper[1].data['DETE']
                if self._det != "singles":
                    time_sgl = time_sgl[dets_sgl == self._det]
                    energy_sgl = hdu_oper[1].data['energy'][dets_sgl ==
                                                            self._det]
                else:
                    energy_sgl = hdu_oper[1].data['energy']

                dets_psd = hdu_oper[2].data['DETE']
                if self._det != "singles":
                    time_psd = time_psd[dets_psd == self._det]
                    energy_psd = hdu_oper[2].data['energy'][dets_psd ==
                                                            self._det]
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

        if self._det in range(19):

            self._times = time_psd
            self._energies = energy_psd

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

            raise AssertionError(f"The detector {self._det} has zero counts"
                                 " and is therefore not active."
                                 " Please exclude this detector!")

        # Bin this in the energy bins we have
        self._energy_bins = np.ones_like(self._energies, dtype=int)*-1
        # Loop over ebins
        for i, (emin, emax) in enumerate(zip(self._ebounds[:-1],
                                             self._ebounds[1:])):
            mask = np.logical_and(self._energies > emin,
                                  self._energies < emax)
            self._energy_bins[mask] = np.ones_like(self._energy_bins[mask])*i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        mask = self._energy_bins == -1
        self._energy_bins = self._energy_bins[~mask]
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]

    def _get_deadtime_info(self):
        """
        Get the deadtime info from the hk file

        :returns:
        """

        # read in the OB-Time (on-board time) and the ISDC-MJD times
        # of the events from the data file. We will need this to
        # convert the OB-Times given in the hk file into ISDC-MJD time
        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data',
                                    self._pointing_id,
                                    'spi_oper.fits.gz')) as hdu_oper:

            ob_time = hdu_oper["SPI.-OSGL-ALL"].data["OB_TIME"]
            time = hdu_oper["SPI.-OSGL-ALL"].data["TIME"]

        # calulate the true on-board time from the 4 given values
        # in the fits files. See:
        # (https://www.isdc.unige.ch/integral/support/faq.cgi?DATA-006)

        ob_time_data = (ob_time[:, 0]*655363**3 +
                        ob_time[:, 1]*655362**2 +
                        ob_time[:, 2]*65536 +
                        ob_time[:, 3])

        # read in hk file with the deadtime per 1 second intervall
        # in units of 100 nano seconds
        deadtimes = np.array([])
        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data',
                                    self._pointing_id,
                                    'spi_science_hk.fits.gz')) as hdu_oper:

            for i in range(19):
                deadtimes = np.append(deadtimes,
                                      hdu_oper["SPI.-SCHK-HRW"].
                                      data[f"P__DF__CAFDT__L{i}"])

            ob_time = hdu_oper["SPI.-SCHK-CNV"].data["OB_TIME"]

        # reshape to get shape (num_det, num_timebins)
        # and multipy by 100*10**-9 to get the deadtime in
        # seconds
        deadtimes = deadtimes.reshape(19, -1)*100*10**-9

        # again get true OB-Time
        ob_time_hk = (ob_time[:, 0]*655363**3 +
                      ob_time[:, 1]*655362**2 +
                      ob_time[:, 2]*65536 +
                      ob_time[:, 3])

        # use linear interpolation of OB-Time/ISDC-MJD pairs from
        # the data file to get the ISDC-MJD time of the hk time bins
        times_hk = np.interp(ob_time_hk, ob_time_data, time)

        # add end of last time bin (+1 second in units of days)
        times_hk = np.append(times_hk, times_hk[-1]+(1.0/(24*3600)))

        # save time and deadtime - add end of
        self._deadtime_bin_edges = ISDC_MJD_to_cxcsec(times_hk) - \
            self._norm_time
        self._deadtimes = deadtimes[self._det]


    @property
    def geometry_file_path(self):
        """
        :returns: Path to the spacecraft geometry file
        """
        return os.path.join(get_path_of_external_data_dir(), 'pointing_data',
                            self._pointing_id, 'sc_orbit_param.fits.gz')

    @property
    def times(self):
        """
        :returns: times of detected events
        """
        return self._times

    @property
    def energies(self):
        """
        :returns: energies of detected events
        """
        return self._energies

    @property
    def energy_bins(self):
        """
        :returns: energy bin number of every event
        """
        return self._energy_bins

    @property
    def ebounds(self):
        """
        :returns: ebounds of analysis
        """
        return self._ebounds

    @property
    def det(self):
        """
        :returns: detector ID
        """
        return self._det

    @property
    def n_channels(self):
        """
        :returns: number energy channels
        """
        return self._n_channels

    @property
    def time_start(self):
        """
        :returns: start time of lightcurve
        """
        return self._time_start

    @property
    def time_stop(self):
        """
        :returns: stop time of lightcurve
        """
        return self._time_stop

    @property
    def det_name(self):
        """
        :returns: Name det
        """
        return self._det_name

    @property
    def mission(self):
        """
        :returns: Name Mission
        """
        return self._mission

    @property
    def deadtime_bin_starts(self):
        """
        :returns: Start time of time bins which have the deadtime
        information
        """
        return self._deadtime_bin_edges[:-1]

    @property
    def deadtime_bin_stops(self):
        """
        :returns: Stop time of time bins which have the deadtime
        information
        """
        return self._deadtime_bin_edges[1:]

    @property
    def deadtimes_per_interval(self):
        """
        :returns: Deadtime per time bin which have the deadtime
        information
        """
        return self._deadtimes

    @property
    def livetimes_per_interval(self):
        """
        :returns: Livetime per time bin which have the deadtime
        information
        """
        return 1-self._deadtimes


class SPISWFileGRB(object):

    def __init__(self, det, ebounds, time_of_grb, sgl_type=None):
        """
        Class to read in all the data needed from a SCW file for a given
        grbtime

        :param det: For which detector?
        :param ebounds: Ebounds for the Analysis.
        :param time_of_grb: Time of the GRB as "YYMMDD HHMMSS"
        :param sgl_type: Which type of single events?
        Only normal sgl, psd or both?

        :returns: Object
        """

        self._det_name = f"Detector {det}"
        self._mission = "Integral/SPI"

        # Set ebounds of energy bins
        self._ebounds = ebounds

        # How many echans?
        self._n_channels = len(self._ebounds)-1

        # Time of GRB. Needed to get the correct pointing.
        self._time_of_GRB = get_time_object(time_of_grb)

        # Check that det is a valid number
        if not isinstance(det, list):
            self._det = np.array([det])
        else:
            self._det = np.array(det)

        mask = []
        live_dets = get_live_dets(time_of_grb)
        for d in self._det:
            assert d in np.arange(85), f"{d} is not a valid"\
                " detector. Please only use detector ids between 0 and 84."

            if d not in live_dets:
                print(f"Detector {d} removed because it was not working.")
                mask.append(False)
            else:
                mask.append(True)
        self._det = self._det[mask]

        if np.any(self._det < 19):
            assert sgl_type is not None, "Only PSD Events?"
            assert sgl_type in ["psd", "sgl", "both"], \
                "sgl_type must be psd, sgl or both"
            self._sgl_type = sgl_type
        # Find the SW ID of the data file we need for this time
        self._pointing_id = find_needed_ids(self._time_of_GRB)

        # Get the data, either from afs or from ISDC archive
        try:
            # Get the data from the afs server
            get_files(self._pointing_id, access="afs")
        except AssertionError:
            # Get the files from the iSDC data archive
            print("AFS data access did not work."
                  " I will try the ISDC data archive.")
            get_files(self._pointing_id, access="isdc")

        # Reference time of GRB
        self._GRB_ref_time_cxcsec =\
            ISDC_MJD_to_cxcsec((self._time_of_GRB).tt.mjd-51544)
        # Read in all we need
        self._read_in_pointing_data()
        self._get_deadtime_info()
        self._get_gti_info()
        self._apply_gti_to_events()

        self._apply_gti_to_live_time()

    def _read_in_pointing_data(self):
        """
        Gets all needed information from the data file for the given
        pointing_id

        :param pointing_id: pointing_id for which we want the data

        :returns:
        """

        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data', self._pointing_id,
                                    'spi_oper.fits.gz')) as hdu_oper:

            # Get time of first and last event (t0 at grb time)
            time_sgl = ISDC_MJD_to_cxcsec(hdu_oper[1].data['time']) \
                - self._GRB_ref_time_cxcsec
            time_psd = ISDC_MJD_to_cxcsec(hdu_oper[2].data['time']) \
                - self._GRB_ref_time_cxcsec
            time_me2 = ISDC_MJD_to_cxcsec(hdu_oper[4].data['time']) \
                - self._GRB_ref_time_cxcsec
            time_me3 = ISDC_MJD_to_cxcsec(hdu_oper[5].data['time']) \
                - self._GRB_ref_time_cxcsec

            # Get time of first and last detected photon
            self._time_start = np.min(np.concatenate([time_sgl, time_psd,
                                                      time_me2, time_me3]))
            self._time_stop = np.max(np.concatenate([time_sgl, time_psd,
                                                     time_me2, time_me3]))

            # Read in the data for the wanted detector
            # For single events we have to take both the non_psd
            # (often called sgl here...) and the psd events.
            # Both added together give the real single events.
            self._times = np.array([])
            self._energies = np.array([])

            for d in self._det:
                #TODO refactor this

                if d in range(19) or d == -1:
                    if self._sgl_type == "psd" or self._sgl_type == "both":
                        dets_sgl = hdu_oper[1].data['DETE']
                        self._times = np.append(self._times,
                                                time_sgl[dets_sgl == d])
                        self._energies = np.append(self._energies,
                                                   hdu_oper[1].data['energy'][dets_sgl == d])

                    if self._sgl_type == "sgl" or self._sgl_type == "both":
                        dets_psd = hdu_oper[2].data['DETE']
                        self._times = np.append(self._times,
                                                time_psd[dets_psd == d])
                        self._energies = np.append(self._energies,
                                                   hdu_oper[2].data['energy'][dets_psd == d])

                if d in range(19, 61):
                    dets_me2 = np.sort(hdu_oper[4].data['DETE'], axis=1)
                    i, k = double_names[d]
                    mask = np.logical_and(dets_me2[:, 0] == i,
                                          dets_me2[:, 1] == k)

                    self._times = np.append(self._times, time_me2[mask])
                    self._energies = np.append(self._energies,
                                               np.sum(hdu_oper[4].data['energy'][mask], axis=1))

                if d in range(61, 85):
                    dets_me3 = np.sort(hdu_oper[5].data['DETE'], axis=1)
                    i, j, k = triple_names[d]
                    mask = np.logical_and(np.logical_and(dets_me3[:, 0] == i,
                                                         dets_me3[:, 1] == j),
                                          dets_me3[:, 2] == k)

                    self._times = np.append(self._times, time_me3[mask])
                    self._energies = np.append(self._energies,
                                               np.sum(hdu_oper[5].data['energy'][mask], axis=1))

            # sort in time
            sort_array = np.argsort(self._times)
            self._times = self._times[sort_array]
            self._energies = self._energies[sort_array]

        # Check if there are any counts
        if np.sum(self._energies) == 0:

            raise AssertionError(f"The detector {self._det} has zero counts "
                                 "and is therefore not active. "
                                 "Please exclude this detector!")

        # Assign the corresponsing energy channel number to every detected
        # energy
        self._energy_bins = np.ones_like(self._energies, dtype=int)*-1
        # Loop over ebins
        for i, (emin, emax) in enumerate(zip(self._ebounds[:-1],
                                             self._ebounds[1:])):
            mask = np.logical_and(self._energies > emin,
                                  self._energies < emax)
            self._energy_bins[mask] = np.ones_like(self._energy_bins[mask])*i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        mask = self._energy_bins == -1
        self._energy_bins = self._energy_bins[~mask]
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]

    def _get_gti_info(self):
        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data',
                                    self._pointing_id,
                                    'spi_gti.fits.gz')) as hdu_oper:

            self._gti_start = (ISDC_MJD_to_cxcsec(hdu_oper[2].data["START"]) -
                               self._GRB_ref_time_cxcsec)
            self._gti_stop = (ISDC_MJD_to_cxcsec(hdu_oper[2].data["STOP"]) -
                              self._GRB_ref_time_cxcsec)

        
    def _get_deadtime_info(self):
        """
        Get the deadtime info from the hk file

        :returns:
        """

        # read in the OB-Time (on-board time) and the ISDC-MJD times
        # of the events from the data file. We will need this to
        # convert the OB-Times given in the hk file into ISDC-MJD time
        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data',
                                    self._pointing_id,
                                    'spi_oper.fits.gz')) as hdu_oper:

            ob_time = hdu_oper["SPI.-OSGL-ALL"].data["OB_TIME"]
            time = hdu_oper["SPI.-OSGL-ALL"].data["TIME"]

        # calulate the true on-board time from the 4 given values
        # in the fits files. See:
        # (https://www.isdc.unige.ch/integral/support/faq.cgi?DATA-006)

        ob_time_data = (ob_time[:, 0]*655363**3 +
                        ob_time[:, 1]*655362**2 +
                        ob_time[:, 2]*65536 +
                        ob_time[:, 3])

        # read in hk file with the deadtime per 1 second intervall
        # in units of 100 nano seconds
        deadtimes = np.array([])
        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data',
                                    self._pointing_id,
                                    'spi_science_hk.fits.gz')) as hdu_oper:

            for i in range(19):
                deadtimes = np.append(deadtimes,
                                      hdu_oper["SPI.-SCHK-HRW"].
                                      data[f"P__DF__CAFDT__L{i}"])

            ob_time = hdu_oper["SPI.-SCHK-CNV"].data["OB_TIME"]

        # reshape to get shape (num_det, num_timebins)
        # and multipy by 100*10**-9 to get the deadtime in
        # seconds
        deadtimes = deadtimes.reshape(19, -1)*100*10**-9

        # Sometimes there are deadtimes>1 in the file, because well...
        # We have to correct this because this will lead to negative
        # exposure times... Just replace it with the closes value to the
        # left in the array that is smaller than 1 and pray
        # that this is not also wrong...

        for d in range(19):
            idx = np.argwhere(deadtimes[d] > 1).flatten()
            if len(idx) > 0:
                # we have unvalid deadtimes
                for i in idx:
                    subarray = deadtimes[d, :i:-1]
                    val = subarray[subarray < 1].flatten()[0]
                    deadtimes[d, i] = val

        assert np.all(deadtimes < 1), "Deadtimes larger than 1 are not possible."

        # again get true OB-Time
        ob_time_hk = (ob_time[:, 0]*655363**3 +
                      ob_time[:, 1]*655362**2 +
                      ob_time[:, 2]*65536 +
                      ob_time[:, 3])

        # use linear interpolation of OB-Time/ISDC-MJD pairs from
        # the data file to get the ISDC-MJD time of the hk time bins
        times_hk = np.interp(ob_time_hk, ob_time_data, time)

        # add end of last time bin (+1 second in units of days)
        times_hk = np.append(times_hk, times_hk[-1]+(1.0/(24*3600)))

        # save time and deadtime - add end of
        self._deadtime_bin_edges = ISDC_MJD_to_cxcsec(times_hk) - \
            self._GRB_ref_time_cxcsec

        #if self._det == -1:
        #    self._deadtimes = np.mean(deadtimes, axis=0)
        #else:
        self._deadtimes = np.mean(deadtimes[self._det], axis=0)

        """
        # check if deadtime bin edges cover all the arrival times
        if self._times[0] < self._deadtime_bin_edges[0]:
            # lower end not covered...
            # TODO Add warning here

            # use mean deadtime rate from rest of pointing
            add_deadtime = (np.sum(self._deadtimes) /
                            (self._deadtime_bin_edges[-1] -
                             self._deadtime_bin_edges[0]) *
                            (self._deadtime_bin_edges[0] -
                             self._times[0]))

            self._deadtime_bin_edges =\
                np.insert(self._deadtime_bin_edges, 0, self._times[0])

            self._deadtimes =\
                np.insert(self._deadtimes, 0, add_deadtime)

        if self._times[-1] > self._deadtime_bin_edges[0]:
            # upper end not covered...
            # TODO Add warning here

            # use mean deadtime rate from rest of pointing
            add_deadtime = (np.sum(self._deadtimes) /
                            (self._deadtime_bin_edges[-1] -
                             self._deadtime_bin_edges[0]) *
                            (self._times[-1] -
                             self._deadtime_bin_edges[-1]))

            self._deadtime_bin_edges =\
                np.append(self._deadtime_bin_edges, self._times[-1])

            self._deadtimes =\
                np.append(self._deadtimes, add_deadtime)
        """
        # check if deadtime bin edges cover all the arrival times
        # cutoff data otherwise
        mask = np.logical_or(self._times < self._deadtime_bin_edges[0],
                             self._times > self._deadtime_bin_edges[-1])
        self._times = self._times[~mask]
        self._energies = self._energies[~mask]
        self._energy_bins = self._energy_bins[~mask]

        # always one second intervals in total
        self._livetimes = 1-self._deadtimes

    def _add_time_bins(self, starts, stops, max_time=1):

        save_livetimes = copy.deepcopy(self._livetimes)
        for i, (start, stop) in enumerate(zip(starts,
                                              stops)):
            if stop-start < max_time or max_time==-1:
                # find the deadtime interval we need to split
                # -0.001 for numerical problems
                idx_start = np.argwhere(start >= self._deadtime_bin_edges)[-1][0]
                idx_stop = np.argwhere(stop > self._deadtime_bin_edges)[-1][0]
                if idx_start == idx_stop:
                    # case A gti interval within one deadtime interval

                    timebin_width = (self._deadtime_bin_edges[idx_start+1] -
                                     self._deadtime_bin_edges[idx_start])

                    self._deadtime_bin_edges =\
                        np.insert(self._deadtime_bin_edges, idx_start+1, stop)

                    self._deadtime_bin_edges =\
                        np.insert(self._deadtime_bin_edges, idx_start+1, start)

                    livetime_rate_tot = self._livetimes[idx_start] / timebin_width
                    self._livetimes[idx_start] =\
                        livetime_rate_tot*(self._deadtime_bin_edges[idx_start+1] -
                                      self._deadtime_bin_edges[idx_start])

                    self._livetimes =\
                        np.insert(self._livetimes,
                                  idx_start+1,
                                  livetime_rate_tot*(self._deadtime_bin_edges[idx_start+3] -
                                                     self._deadtime_bin_edges[idx_start+2]))

                    self._livetimes =\
                        np.insert(self._livetimes,
                                  idx_start+1,
                                  livetime_rate_tot*(self._deadtime_bin_edges[idx_start+2] -
                                                self._deadtime_bin_edges[idx_start+1]))
                elif idx_start+1 == idx_stop:
                    # case B gti interval within two deadtime intervals

                    timebin_width1 = (self._deadtime_bin_edges[idx_start+1] -
                                      self._deadtime_bin_edges[idx_start])
                    timebin_width2 = (self._deadtime_bin_edges[idx_start+2] -
                                      self._deadtime_bin_edges[idx_start+1])
                    old_edge = self._deadtime_bin_edges[idx_start+1]
                    self._deadtime_bin_edges[idx_start+1] = start
                    self._deadtime_bin_edges =\
                        np.insert(self._deadtime_bin_edges, idx_start+2, stop)

                    livetime_rate_tot1 = self._livetimes[idx_start]/timebin_width1
                    livetime_rate_tot2 = self._livetimes[idx_start+1]/timebin_width2

                    self._livetimes[idx_start] =\
                        livetime_rate_tot1*(self._deadtime_bin_edges[idx_start+1] -
                                            self._deadtime_bin_edges[idx_start])
                    self._livetimes[idx_start+1] =\
                        livetime_rate_tot2*(self._deadtime_bin_edges[idx_start+3] -
                                            self._deadtime_bin_edges[idx_start+2])
                    self._livetimes =\
                        np.insert(self._livetimes,
                                  idx_start+1,
                                  (livetime_rate_tot1 *
                                   (old_edge -
                                    self._deadtime_bin_edges[idx_start+1]) +
                                   livetime_rate_tot2 *
                                   (self._deadtime_bin_edges[idx_start+2] -
                                    old_edge)))
                else:
                    # case C covers several time bins
                    # merge all together to one bin

                    # first the start edge
                    timebin_width1 = (self._deadtime_bin_edges[idx_start+1] -
                                      self._deadtime_bin_edges[idx_start])
                    livetime_rate_tot1 = self._livetimes[idx_start]/timebin_width1
                    new_livetime1 =\
                        livetime_rate_tot1*(start-self._deadtime_bin_edges[idx_start])

                    # last bin
                    if idx_stop+1 == len(self._deadtime_bin_edges):
                        new_livetimelast = 0
                         # sum the rest
                        new_livetime_center = self._livetimes[idx_start]-new_livetime1
                        for idx in range(idx_start+1, idx_stop):
                            new_livetime_center += self._livetimes[idx]
                        idx_stop -= 1
                    else:
                        timebin_widthlast = (self._deadtime_bin_edges[idx_stop+1] -
                                          self._deadtime_bin_edges[idx_stop])
                        livetime_rate_totlast = self._livetimes[idx_stop]/timebin_widthlast
                        new_livetimelast =\
                            livetime_rate_totlast*(self._deadtime_bin_edges[idx_stop+1]-stop)

                        # sum the rest
                        new_livetime_center = (self._livetimes[idx_start]-new_livetime1+
                                               self._livetimes[idx_stop]-new_livetimelast)
                        for idx in range(idx_start+1, idx_stop):
                            new_livetime_center += self._livetimes[idx]

                    # delete old time bins from arrays
                    self._livetimes = np.delete(self._livetimes,
                                                range(idx_start, idx_stop+1))
                    self._deadtime_bin_edges =\
                        np.delete(self._deadtime_bin_edges,
                                    range(idx_start+1, idx_stop+1))

                    # add new ones
                    self._livetimes = np.insert(self._livetimes,
                                                idx_start,
                                                [new_livetime1,
                                                 new_livetime_center,
                                                 new_livetimelast])
                    self._deadtime_bin_edges =\
                        np.insert(self._deadtime_bin_edges,
                                  idx_start+1,
                                  [start, stop])


                #else:
                #    # merge the two bins
                #    self._deadtime_bin_edges = np.delete(
                #        self._deadtime_bin_edges, idx_start+1
                #    )

                #    new_livetime = self._livetimes[idx_start] + self._livetimes[idx_start+1]

                #    self._livetimes = np.delete(
                #        self._livetimes, idx_start+1
                #    )

                #    self._livetimes[idx_start] = new_livetime



                # remove double entries
                mask =\
                    (self._deadtime_bin_edges[1:] -
                     self._deadtime_bin_edges[:-1]) > 0

                deadtime_bin_edges_inter = self._deadtime_bin_edges[:-1][mask]

                self._deadtime_bin_edges = np.append(deadtime_bin_edges_inter,
                                                     self._deadtime_bin_edges[-1])

                self._livetimes = self._livetimes[mask]

        assert np.isclose(np.sum(self._livetimes), np.sum(save_livetimes))

    def _apply_gti_to_live_time(self):

        # first check if there are so small gti and bti that we should add this to the
        # deadtime bins

        # first gti
        self._add_time_bins(self._gti_start, self._gti_stop)

        # now bti (times between gti)
        bti_start = self._gti_stop[:-1]
        bti_stop = self._gti_start[1:]
        self._add_time_bins(bti_start, bti_stop, max_time=-1)
        #return self._livetimes
        # loop throught the dead time intervals
        for i, (start, stop) in enumerate(zip(self._deadtime_bin_edges[:-1],
                                              self._deadtime_bin_edges[1:])):

            # check if completly inside one of the gti intervals
            idx = np.argwhere(start >= self._gti_start)[-1][0]
            idx_stop = np.argwhere(stop >= self._gti_start)[-1][0]

            if start > self._gti_stop[idx]:
                # case A the whole dead time interval is outside
                # the gti intervals
                frac_good = 0

            elif self._gti_stop[idx] >= stop:
                # case B whole interval in one gti interval
                frac_good = 1

            elif idx_stop == idx:
                # case C dead time interval intersects with only
                # one gti interval
                frac_good = (self._gti_stop[idx]-start)/(stop-start)

            elif idx_stop == idx+1:
                # case D dead time interval intersect with exactly two gti

                # gti 1
                frac_good = (self._gti_stop[idx]-start)/(stop-start)

                # gti 2
                frac_good += (stop-self._gti_start[idx_stop])/(stop-start)

            else:
                # case E dead time interval intersect with more than two gti

                # first gti
                frac_good = (self._gti_stop[idx]-start)/(stop-start)

                # last gti
                frac_good += (stop-self._gti_start[idx_stop])/(stop-start)

                # all gti inbetween

                for idx_loop in range(idx+1, idx_stop):
                    frac_good += (self._gti_stop[idx_loop] -
                                  self._gti_start[idx_loop])/(stop-start)

            self._livetimes[i] *= frac_good

        return self._livetimes

    def _apply_gti_to_events(self):
        """
        This created a filter index for events falling outside of the
        GTI. It must be run after the events are binned in energy because
        a filter is set up in that function for events that have energies
        outside the EBOUNDS of the DRM

        Taken from threeML lat_data.py
        :return: none
        """

        # initial filter
        filter_idx = np.zeros_like(self._times, dtype=bool)

        # loop throught the GTI intervals
        for start, stop in zip(self._gti_start, self._gti_stop):

            # capture all the events within that interval
            tmp_idx = np.logical_and(start <= self._times, self._times <= stop)

            # combine with the already selected events
            filter_idx = np.logical_or(filter_idx, tmp_idx)

        self._filter_idx = filter_idx

    @property
    def geometry_file_path(self):
        """
        :returns: Path to the spacecraft geometry file
        """
        return os.path.join(get_path_of_external_data_dir(),
                            'pointing_data',
                            self._pointing_id,
                            'sc_orbit_param.fits.gz')

    @property
    def gti_times(self):
        """
        :returns: times of detected events within a gti
        """
        return self._times[self._filter_idx]

    @property
    def gti_energies(self):
        """
        :returns: energies of detected events
        """
        return self._energies[self._filter_idx]

    @property
    def gti_energy_bins(self):
        """
        :returns: energy bin number of every event
        """
        return self._energy_bins[self._filter_idx]

    @property
    def times(self):
        """
        :returns: times of detected events
        """
        return self._times

    @property
    def energies(self):
        """
        :returns: energies of detected events
        """
        return self._energies

    @property
    def energy_bins(self):
        """
        :returns: energy bin number of every event
        """
        return self._energy_bins

    @property
    def ebounds(self):
        """
        :returns: ebounds of analysis
        """
        return self._ebounds

    @property
    def det(self):
        """
        :returns: detector ID
        """
        return self._det

    @property
    def n_channels(self):
        """
        :returns: number energy channels
        """
        return self._n_channels

    @property
    def time_start(self):
        """
        :returns: start time of lightcurve
        """
        return self._time_start

    @property
    def time_stop(self):
        """
        :returns: stop time of lightcurve
        """
        return self._time_stop

    @property
    def det_name(self):
        """
        :returns: Name det
        """
        return self._det_name

    @property
    def mission(self):
        """
        :returns: Name Mission
        """
        return self._mission

    @property
    def deadtime_bin_starts(self):
        """
        :returns: Start time of time bins which have the deadtime
        information
        """
        return self._deadtime_bin_edges[:-1]

    @property
    def deadtime_bin_stops(self):
        """
        :returns: Stop time of time bins which have the deadtime
        information
        """
        return self._deadtime_bin_edges[1:]

    #@property
    #def deadtimes_per_interval(self):
    #    """
    #    :returns: Deadtime per time bin which have the deadtime
    #    information
    #    """
    #    return self._deadtimes

    @property
    def livetimes_per_interval(self):
        """
        :returns: Livetime per time bin which have the deadtime
        information
        """
        return self._livetimes #1-self._deadtimes


class TimeSeriesBuilderSPI(TimeSeriesBuilder):

    def __init__(
        self,
        name,
        time_series,
        response=None,
        poly_order=-1,
        verbose=True,
        restore_poly_fit=None,
        container_type=BinnedSpectrumWithDispersion,
        **kwargs
    ):
        """
        Class to build the time_series for SPI. Inherited from the 3ML
        TimeSeriesBuilder with added class methods to build the object
        for SPI datafiles.
        :param name: Name of the tsb
        :param time_series: Timeseries with the data
        :param response: Response object
        :param poly_order: poly order for the polynominal fitting
        :param verbose: Verbose?
        :param restore_poly_fit: Path to a file with the poly bkg fits
        :param containter_type: ContainerType for spectrum
        :returns: Object
        """

        super(TimeSeriesBuilderSPI, self).__init__(
            name,
            time_series,
            response=response,
            poly_order=poly_order,
            unbinned=False,
            verbose=verbose,
            restore_poly_fit=restore_poly_fit,
            container_type=container_type,
            **kwargs
        )

    @classmethod
    def from_spi_grb(cls,
                     name,
                     det,
                     time_of_grb,
                     ebounds=None,
                     response=None,
                     sgl_type=None,
                     restore_background=None,
                     poly_order=0,
                     verbose=True
    ):
        """
        Class method to build the time_series_builder for a given GRB time

        :param name: Name of object
        :param det: Which det?
        :param ebounds: Output ebounds for analysis.
        :param time_of_grb: Astropy time object with the time of the GRB (t0)
        :param response: InstrumenResponse Object
        :param sgl_type: What kind of sinlge events? Standard single events?
                         PSD events? Or both?
        :param restore_background: File to restore bkg
        :param poly_order: Which poly_order? -1 gives automatic determination
        :param verbose: Verbose?

        :returns: Initalized TimeSeriesBuilder object
        """

        assert ebounds is not None or response is not None, "You have to "\
            "either specify ebounds or input a response object."

        if ebounds is not None and response is not None:
            # check that the ebounds match
            assert np.all(response.ebounds == ebounds), "Ebounds do not "\
                "match the ones in the response object!"

        elif response is not None:
            # ebounds is None in this case
            ebounds = response.ebounds

        # Get an object with all the needed information
        spi_grb_setup = SPISWFileGRB(det,
                                     ebounds,
                                     time_of_grb,
                                     sgl_type=sgl_type)

        event_list = EventListWithLiveTime(
            arrival_times=spi_grb_setup.gti_times,
            measurement=spi_grb_setup.gti_energy_bins,
            n_channels=spi_grb_setup.n_channels,
            start_time=spi_grb_setup.time_start,
            stop_time=spi_grb_setup.time_stop,
            live_time=spi_grb_setup.livetimes_per_interval,
            live_time_starts=spi_grb_setup.deadtime_bin_starts,
            live_time_stops=spi_grb_setup.deadtime_bin_stops,
            first_channel=0,
            instrument=spi_grb_setup.det_name,
            mission=spi_grb_setup.mission,
            verbose=verbose,
            edges=spi_grb_setup.ebounds,
        )

        return cls(
            name,
            event_list,
            poly_order=poly_order,
            verbose=verbose,
            restore_poly_fit=restore_background,
            response=response,
            container_type=BinnedSpectrumWithDispersion
        )

    @classmethod
    def from_spi_constant_pointing(cls,
                                   det=0,
                                   pointing_id="118900570010",
                                   ebounds=None,
                                   response=None,
    ):
        """
        Class method to build the time_series_builder for a given pointing id

        :param det: Which det?
        :param ebounds: Output ebounds for analysis.
        :param pointing_id: Pointing ID
        :param response: InstrumenResponse Object

        :returns: Initalized TimeSeriesBuilder object
        """

        assert ebounds is not None or response is not None, "You have to "\
            "either specify ebounds or input a response object."

        if ebounds is not None and response is not None:
            # check that the ebounds match
            assert np.all(response.ebounds == ebounds), "Ebounds do not "\
                "match the ones in the response object!"

        elif response is not None:
            # ebounds is None in this case
            ebounds = response.ebounds

        spi_grb_setup1 = SPISWFile(det, pointing_id, ebounds)

        event_list = EventListWithLiveTime(
            arrival_times=spi_grb_setup1.times,
            measurement=spi_grb_setup1.energy_bins,
            n_channels=spi_grb_setup1.n_channels,
            start_time=spi_grb_setup1.time_start,
            stop_time=spi_grb_setup1.time_stop,
            live_time=spi_grb_setup1.livetimes_per_interval,
            live_time_starts=spi_grb_setup1.deadtime_bin_starts,
            live_time_stops=spi_grb_setup1.deadtime_bin_stops,
            first_channel=0,
            instrument=spi_grb_setup1.det_name,
            mission=spi_grb_setup1.mission,
            verbose=False,
            edges=spi_grb_setup1.ebounds,
        )

        tsb = cls(f"SPIID{pointing_id}D{det}",
                  event_list,
                  verbose=True,
                  response=response,
                  container_type=BinnedSpectrumWithDispersion)

        # set active time to total time of sw
        tsb.set_active_time_interval(f"{spi_grb_setup1.times[0]}-"
                                     f"{spi_grb_setup1.times[-1]}")

        return tsb
