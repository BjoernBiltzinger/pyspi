from pyspi.utils.livedets import get_live_dets
from pyspi.utils.response.spi_pointing import _construct_sc_matrix, \
    _transform_spi_to_icrs, SPIPointing
from pyspi.utils.response.spi_response import ResponsePhotopeak, \
    multi_response_irf_read_objects
from pyspi.utils.detector_ids import double_names, triple_names
from pyspi.utils.function_utils import ISDC_MJD_to_cxcsec, \
    construct_energy_bins, find_needed_ids
from pyspi.io.package_data import get_path_of_external_data_dir
from pyspi.io.get_files import get_files_afs, get_files_isdcarc

import numpy as np
from datetime import datetime
from astropy.time.core import Time
from astropy.io import fits
import os
from threeML.utils.spectrum.binned_spectrum import BinnedSpectrum
from threeML.utils.spectrum.binned_spectrum_set import BinnedSpectrumSet
from threeML.utils.time_interval import TimeIntervalSet
class SimulateGRBDataFile(object):

    def __init__(self, model, time_of_GRB, ebounds,
                 norm_function=None, az=0, zen=0, psd_eff=0.86):
        """
        Init simulation object
        :param model: Astromodel with spectral form at maximum
        :param norm_function: norm(t) with values between 0 and 1
        :param time_of_GRB: start time of GRB - Needed for correct response version
        :param az: Azimuth pos of GRB in sat. frame in deg
        :param zen: Zenith pos of GRB in sat. frame in deg
        """
        if norm_function is None:
            def norm_function_default(t):
                tpeak = 10
                mask_rise = np.logical_and(t>0, t<=tpeak)
                mask_decay = t>tpeak
                res = np.zeros(len(t))
                res[mask_rise] = t[mask_rise]**3/tpeak**3
                res[mask_decay] = (tpeak)*(t[mask_decay])**-1
                return res
            norm_function = norm_function_default

        # Make sure no ebounds outside and inside of electronic noise area...
        self._ebounds = ebounds
        self._ebounds, self._non_psd_mask = construct_energy_bins(self._ebounds)

        # Build psd_eff area to reduce response in the electronic noise area, as only
        # these events will be used...
        self._psd_eff = psd_eff
        self._eff_area_corr = np.ones(len(self._ebounds)-1)
        self._eff_area_corr[~self._non_psd_mask] = self._psd_eff
        
        self._model = model
        self._str_time = time_of_GRB
        time = datetime.strptime(time_of_GRB, '%y%m%d %H%M%S')
        self._time_of_GRB = Time(time)

        self._pointing_id = find_needed_ids(self._time_of_GRB)
        # Get the data, either from afs or from ISDC archive
        try:
            # Get the data from the afs server
            get_files_afs(self._pointing_id)
        except:
            # Get the files from the iSDC data archive
            print('AFS data access did not work. I will try the ISDC data archive.')
            get_files_isdcarc(self._pointing_id)

        geometry_file_path = os.path.join(get_path_of_external_data_dir(),
                                          'pointing_data',
                                          self._pointing_id,
                                          'sc_orbit_param.fits.gz')

        pointing_object = SPIPointing(geometry_file_path)
        self._sc_matrix = _construct_sc_matrix(**pointing_object.sc_points[10])

        self._ra, self._dec = _transform_spi_to_icrs(az, zen, self._sc_matrix)

        self._norm_function = norm_function

        self._live_dets = get_live_dets(time=self._str_time,
                                        event_types=["single", "double",
                                                     "triple"])

        # SPI aligned with icrs frame


        self._time_bin_bounds = np.arange(-200, 200, 1.)
        self._time_intervals = TimeIntervalSet.from_starts_and_stops(self._time_bin_bounds[:-1],
                                                                     self._time_bin_bounds[1:])
        self._read_in_example_background()

        # Get the response objects
        self._binned_spectra_set = {}
        max_flux = model.get_point_source_fluxes(0, self._ebounds)
        e1, e2 = self._ebounds[:-1], self._ebounds[1:]
        t1, t2 = self._time_bin_bounds[:-1], self._time_bin_bounds[1:]
        for d in self._live_dets:
            response_irf_read_object = multi_response_irf_read_objects([self._time_of_GRB],
                                                                       d,
                                                                       drm="Photopeak")

            rsp = ResponsePhotopeak(self._ebounds, response_irf_read_object[0],
                                    self._sc_matrix, det=d)
            
            rsp.set_location(self._ra, self._dec)

            diffcountrates = norm_function(self._time_bin_bounds)*\
                np.tile(max_flux, (len(self._time_bin_bounds), 1)).T

            res = np.zeros((len(self._ebounds)-1, len(self._time_bin_bounds)))
            for i in range(len(self._ebounds)-1):
                res[i] = np.trapz(diffcountrates[i:i+2].T, self._ebounds[i:i+2])

            res2 = np.zeros((len(self._ebounds)-1, len(self._time_bin_bounds)-1))
            for i in range(len(self._time_bin_bounds)-1):
                res2[:,i] = np.trapz(res[:,i:i+2], self._time_bin_bounds[i:i+2])

            counts = np.random.poisson(self._eff_area_corr*rsp.effective_area*res2.T)
            if d<20:
                # Make check for this electronic noise range... Only
                # Use psd background counts in this range...
                counts[:,self._non_psd_mask] += self._non_psd_bkg[d][:,self._non_psd_mask]
                counts += self._psd_bkg[d]
            elif d<61:
                counts += self._me2_bkg[d]
            else:
                counts += self._me3_bkg[d]
            counts = counts.T
            binned_spectra = []
            for i in range(len(self._time_bin_bounds)-1):
                binned_spectra.append(BinnedSpectrum(
                    counts[:,i],
                    self._time_bin_bounds[i+1]-self._time_bin_bounds[i],
                    self._ebounds,
                    tstart = self._time_bin_bounds[i],
                    tstop = self._time_bin_bounds[i+1],
                    mission="INTEGRAL",
                    instrument=f"SPIDET{d}",
                    is_poisson=True,
                ))
            self._binned_spectra_set[d] = BinnedSpectrumSet(binned_spectra,
                                                            time_intervals=self._time_intervals)

    def _read_in_example_background(self):
        """
        Gets all needed information from the data file for the given pointing_id
        :param pointing_id: pointing_id for which we want the data
        :return:
        """

        with fits.open(os.path.join(get_path_of_external_data_dir(),
                                    'pointing_data',
                                    self._pointing_id,
                                    'spi_oper.fits.gz')) as hdu_oper:


            time_sgl = ISDC_MJD_to_cxcsec(hdu_oper[1].data['time'])
            time_psd = ISDC_MJD_to_cxcsec(hdu_oper[2].data['time'])
            time_me2 = ISDC_MJD_to_cxcsec(hdu_oper[4].data['time'])
            time_me3 = ISDC_MJD_to_cxcsec(hdu_oper[5].data['time'])


            time_start = np.min(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))
            time_stop = np.max(np.concatenate([time_sgl, time_psd, time_me2, time_me3]))

            time_sgl -= time_start+300
            time_psd -= time_start+300
            time_me2 -= time_start+300
            time_me3 -= time_start+300

            time_mask_sgl = np.logical_and(time_sgl>-200, time_sgl<200)
            time_mask_psd = np.logical_and(time_psd>-200, time_psd<200)
            time_mask_me2 = np.logical_and(time_me2>-200, time_me2<200)
            time_mask_me3 = np.logical_and(time_me3>-200, time_me3<200)

            time_sgl = time_sgl[time_mask_sgl]
            time_psd = time_psd[time_mask_psd]
            time_me2 = time_me2[time_mask_me2]
            time_me3 = time_me3[time_mask_me3]

            dets_sgl = hdu_oper[1].data['DETE'][time_mask_sgl]
            energy_sgl = hdu_oper[1].data['energy'][time_mask_sgl]

            dets_psd = hdu_oper[2].data['DETE'][time_mask_psd]
            energy_psd = hdu_oper[2].data['energy'][time_mask_psd]

            dets_me2_raw = np.sort(hdu_oper[4].data['DETE'][time_mask_me2], axis=1)
            dets_me2 = np.ones(len(dets_me2_raw))*-1
            for d in range(19,61):
                i, k = double_names[d]
                mask = np.logical_and(dets_me2_raw[:, 0] == i,
                                      dets_me2_raw[:, 1] == k)
                dets_me2[mask] = d
            mask = dets_me2!=-1
            time_me2 = time_me2[mask]
            energy_me2 = np.sum(hdu_oper[4].data['energy'][time_mask_me2][mask], axis=1)
            dets_me2 = dets_me2[mask]

            dets_me3_raw = np.sort(hdu_oper[5].data['DETE'][time_mask_me3], axis=1)
            dets_me3 = np.ones(len(dets_me3_raw))*-1
            for d in range(61,85):
                i, j, k = triple_names[d]
                mask = np.logical_and(np.logical_and(dets_me3_raw[:, 0] == i,
                                                     dets_me3_raw[:, 1] == j),
                                      dets_me3_raw[:, 2] == k)
                dets_me3[mask] = d
            mask = dets_me3!=-1
            time_me3 = time_me3[mask]
            energy_me3 = np.sum(hdu_oper[5].data['energy'][time_mask_me3][mask], axis=1)
            dets_me3 = dets_me3[mask]
        # Bin this in the energy bins we have
        #energy_bins_non_psd = np.ones_like(energy_sgl, dtype=int)*-1
        # Loop over ebins
        #for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
        #    mask = np.logical_and(energy_sgl>emin, energy_sgl<emax)
        #    energy_bins_non_psd[mask] = i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        #mask = energy_bins_non_psd == -1
        #energy_bins_non_psd = energy_bins_non_psd[~mask]
        #time_non_psd = time_sgl[~mask]
        #dets_non_psd = dets_sgl[~mask]

        # Bin in time
        res = np.zeros((19, len(self._time_bin_bounds)-1, len(self._ebounds)-1))
        for d in range(19):
            mask = dets_sgl == d
            res[d] = np.histogram2d(time_sgl[mask], energy_sgl[mask],
                                    bins=[self._time_bin_bounds, self._ebounds])[0]

        self._non_psd_bkg = np.array(res, dtype=int)

        # Bin this in the energy bins we have
        #energy_bins_psd = np.ones_like(energy_psd, dtype=int)*-1
        # Loop over ebins
        #for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
        #    mask = np.logical_and(energy_psd>emin, energy_psd<emax)
        #    energy_bins_psd[mask] = i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        #mask = energy_bins_psd == -1
        #energy_bins_psd = energy_bins_psd[~mask]
        #time_psd = time_psd[~mask]
        #dets_psd = dets_psd[~mask]
        
        # Bin in time
        res = np.zeros((19, len(self._time_bin_bounds)-1, len(self._ebounds)-1))
        for d in range(19):
            mask = dets_psd == d
            res[d] = np.histogram2d(time_psd[mask], energy_psd[mask],
                                    bins=[self._time_bin_bounds, self._ebounds])[0]

        self._psd_bkg = np.array(res, dtype=int)


        # Bin this in the energy bins we have
        #energy_bins_me2 = np.ones_like(energy_me2, dtype=int)*-1
        # Loop over ebins
        #for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
        #    mask = np.logical_and(energy_me2>emin, energy_me2<emax)
        #    energy_bins_me2[mask] = i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        #mask = energy_bins_me2 == -1
        #energy_bins_me2 = energy_bins_me2[~mask]
        #time_me2 = time_me2[~mask]
        #dets_me2 = dets_me2[~mask]
        # Bin in time
        res = np.zeros((61, len(self._time_bin_bounds)-1, len(self._ebounds)-1))
        for d in range(19,61):
            mask = dets_me2 == d
            res[d] = np.histogram2d(time_me2[mask], energy_me2[mask],
                                    bins=[self._time_bin_bounds, self._ebounds])[0]

        self._me2_bkg = np.array(res, dtype=int)

        # Bin this in the energy bins we have
        #energy_bins_me3 = np.ones_like(energy_me3, dtype=int)*-1
        # Loop over ebins
        #for i, (emin, emax) in enumerate(zip(self._ebounds[:-1], self._ebounds[1:])):
        #    mask = np.logical_and(energy_me3>emin, energy_me3<emax)
        #    energy_bins_me3[mask] = i

        # Throw away all events that have energies outside of the ebounds that
        # should be used
        #mask = energy_bins_me3 == -1
        #energy_bins_me3 = energy_bins_me3[~mask]
        #time_me3 = time_me3[~mask]
        #dets_me3 = dets_me3[~mask]
        # Bin in time
        res = np.zeros((85, len(self._time_bin_bounds)-1, len(self._ebounds)-1))
        for d in range(61,85):
            mask = dets_me3 == d
            res[d] = np.histogram2d(time_me3[mask], energy_me3[mask],
                                    bins=[self._time_bin_bounds, self._ebounds])[0]

        self._me3_bkg = np.array(res, dtype=int)

    def get_binned_spectrum_set(self, det):
        assert det in self.live_dets, "Detector ID not valid or not active."
        return self._binned_spectra_set[det]

    @property
    def live_dets(self):
        return self._live_dets

    @property
    def ra(self):
        return self._ra

    @property
    def dec(self):
        return self._dec
