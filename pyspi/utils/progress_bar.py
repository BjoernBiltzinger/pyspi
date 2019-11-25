import numpy as np
import os
from gbmgeometry import PositionInterpolator, gbm_detector_list
import astropy.time as astro_time

from gbmbkgpy.utils.progress_bar import progress_bar
from gbmbkgpy.io.package_data import get_path_of_external_data_dir
from gbmbkgpy.io.file_utils import file_existing_and_readable
from gbmbkgpy.io.downloading import download_data_file

try:

    # see if we have mpi and/or are upalsing parallel

    from mpi4py import MPI

    if MPI.COMM_WORLD.Get_size() > 1:  # need parallel capabilities
        using_mpi = True  ###################33

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

    else:

        using_mpi = False
except:

    using_mpi = False

valid_det_names = ['n0', 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'na', 'nb']


class Geometry(object):
    def __init__(self, data, det, day_list, n_bins_to_calculate_per_day):
        """
        Initalize the geometry precalculation. This calculates several quantities (e.g. Earth
        position in the satellite frame for n_bins_to_calculate times during the day
        """

        # Test if all the input is valid
        assert type(data.mean_time) == np.ndarray, 'Invalid type for mean_time. Must be an array but is {}.'.format(type(data.mean_time))
        assert det in valid_det_names, 'Invalid det name. Must be one of these {} but is {}.'.format(valid_det_names, det)
        assert type(n_bins_to_calculate_per_day) == int, 'Type of n_bins_to_calculate has to be int but is {}'.format(type(n_bins_to_calculate_per_day))

        # Save everything
        self.mean_time = data.mean_time
        self._det = det
        self._n_bins_to_calculate_per_day = n_bins_to_calculate_per_day
        self._day_start_times = data.day_start_times
        self._day_stop_times = data.day_stop_times
        self._day_list = map(str, sorted(map(int, day_list)))

        # Check if poshist file exists, if not download it and save the paths for all days in an array
        self._pos_hist = np.array([])
        for day in day_list:
            poshistfile_name = 'glg_{0}_all_{1}_v00.fit'.format('poshist', day)
            poshistfile_path = os.path.join(get_path_of_external_data_dir(), 'poshist', poshistfile_name)

            # If using MPI only rank=0 downloads the data, all other have to wait
            if using_mpi:
                if rank == 0:
                    if not file_existing_and_readable(poshistfile_path):
                        download_data_file(day, 'poshist')
                comm.Barrier()
            else:
                if not file_existing_and_readable(poshistfile_path):
                    download_data_file(day, 'poshist')

            # Save poshistfile_path for later usage
            self._pos_hist = np.append(self._pos_hist, poshistfile_path)
        for pos in self._pos_hist:
            assert file_existing_and_readable(pos), '{} does not exist'.format(pos)

        # Number of bins to skip, to equally distribute the n_bins_to_calculate times over the day
        n_skip = int(np.ceil(len(self.mean_time) / (self._n_bins_to_calculate_per_day * len(day_list))))

        # Create the lists of the times where to calculate the geometry
        list_times_to_calculate = self.mean_time[::n_skip]

        # Add start and stop time of days to times for which the geometry should be calculated (to ensure a valid
        # interpolation for all used times
        self._list_times_to_calculate = self._add_start_stop(list_times_to_calculate, self._day_start_times,
                                                             self._day_stop_times)

        # Calculate Geometry. With or without Mpi support.
        for day_number, day in enumerate(day_list):
            if using_mpi:
                sun_angle, sun_positions, time, earth_az, earth_zen, earth_position, quaternion, sc_pos, times_lower_bound_index, times_upper_bound_index = self._one_day_setup_geometery_mpi(day_number)
            else:
                sun_angle, sun_positions, time, earth_az, earth_zen, earth_position, quaternion, sc_pos = \
                    self._one_day_setup_geometery_no_mpi(day_number)
            if day_number == 0:
                self._sun_angle = [sun_angle]
                self._sun_positions = [sun_positions]
                self._time = [time]
                self._earth_az = [earth_az]
                self._earth_zen = [earth_zen]
                self._earth_position = [earth_position]
                self._quaternion = [quaternion]
                self._sc_pos = [sc_pos]
                if using_mpi:
                    self._times_lower_bound_index = np.array([times_lower_bound_index])
                    self._times_upper_bound_index = np.array([times_upper_bound_index])
            else:
                self._sun_angle.append(sun_angle)
                self._sun_positions.append(sun_positions)
                self._time.append(time)
                self._earth_az.append(earth_az)
                self._earth_zen.append(earth_zen)
                self._earth_position.append(earth_position)
                self._quaternion.append(quaternion)
                self._sc_pos.append(sc_pos)
                if using_mpi:
                    self._times_lower_bound_index = np.append(self._times_lower_bound_index, times_lower_bound_index)
                    self._times_upper_bound_index = np.append(self._times_upper_bound_index, times_upper_bound_index)
        self._time = np.concatenate(self._time, axis=0)
        self._sun_positions = np.concatenate(self._sun_positions, axis=0)
        self._sun_angle = np.concatenate(self._sun_angle, axis=0)
        self._earth_az = np.concatenate(self._earth_az, axis=0)
        self._earth_zen = np.concatenate(self._earth_zen, axis=0)
        self._earth_position = np.concatenate(self._earth_position, axis=0)
        self._quaternion = np.concatenate(self._quaternion, axis=0)
        self._sc_pos = np.concatenate(self._sc_pos, axis=0)

    # All properties of the class.
    # Returns the calculated values of the quantities for all the n_bins_to_calculate times
    # Of the day used in setup_geometry
    @property
    def time(self):
        """
        Returns the times of the time bins for which the geometry was calculated
        """

        return self._list_times_to_calculate

    @property
    def time_days(self):
        """
        Returns the times of the time bins for which the geometry was calculated for all days separately as arrays in
        one big array
        """

        return self._time

    @property
    def sun_positions(self):
        """
        :return: sun positions as skycoord object in sat frame for all times for which the geometry was calculated
        """

        return self._sun_positions
    
    @property
    def sun_angle(self):
        """
        Returns the angle between the sun and the line of sight for all times for which the 
        geometry was calculated
        """

        return self._sun_angle

    @property
    def earth_az(self):
        """
        Returns the azimuth angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_az

    @property
    def earth_zen(self):
        """
        Returns the zenith angle of the earth in the satellite frame for all times for which the 
        geometry was calculated
        """

        return self._earth_zen

    @property
    def earth_position(self):
        """
        Returns the Earth position as SkyCoord object for all times for which the geometry was 
        calculated
        """
        return self._earth_position

    @property
    def quaternion(self):
        """
        Returns the quaternions, defining the rotation of the satellite, for all times for which the 
        geometry was calculated
        """

        return self._quaternion

    @property
    def sc_pos(self):
        """
        Returns the spacecraft position, in ECI coordinates, for all times for which the 
        geometry was calculated
        """

        return self._sc_pos

    @property
    def times_upper_bound_index(self):
        """
        Returns the upper time bound of the geometries calculated by this rank
        """
        return self._times_upper_bound_index

    @property
    def times_lower_bound_index(self):
        """
        Returns the lower time bound of the geometries calculated by this rank
        """
        return self._times_lower_bound_index

    def _one_day_setup_geometery_mpi(self, day_number):
        """
        Run the geometry precalculation with mpi support. Only use this funtion if you have MPI
        and are running this on several cores.
        """

        assert using_mpi, 'You need MPI to use this function, please use _setup_geometery_no_mpi if you do not have MPI'

        # Create the PositionInterpolator object with the infos from the poshist file
        position_interpolator = PositionInterpolator(poshist=self._pos_hist[day_number])

        # Init all lists
        sun_angle = []
        sun_positions = []
        time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = []  # earth pos in icrs frame (skycoord)

        # Additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos = []

        # Get the times for which the geometry should be calculated for this day (Build a mask that masks all time bins
        # outside the start and stop day of this time bin

        masksmaller = self._list_times_to_calculate >= self._day_start_times[day_number]
        masklarger = self._list_times_to_calculate <= self._day_stop_times[day_number]

        masktot = masksmaller * masklarger

        list_times_to_calculate = self._list_times_to_calculate[masktot]

        times_per_rank = float(len(list_times_to_calculate)) / float(size)
        times_lower_bound_index = int(np.floor(rank * times_per_rank))
        times_upper_bound_index = int(np.floor((rank + 1) * times_per_rank))

        # Only rank==0 gives some output how much of the geometry is already calculated (progress_bar)
        if rank == 0:
            with progress_bar(len(list_times_to_calculate[times_lower_bound_index:times_upper_bound_index]),
                              title='Calculating geomerty for day {}. This shows the progress of rank 0. '
                                    'All other should be about the same.'.format(self._day_list[day_number])) as p:

                # Calculate the geometry for all times associated with this rank 
                for mean_time in list_times_to_calculate[times_lower_bound_index:times_upper_bound_index]:
                    quaternion_step = position_interpolator.quaternion(mean_time)
                    sc_pos_step = position_interpolator.sc_pos(mean_time)
                    det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                       sc_pos=sc_pos_step,
                                                       time=astro_time.Time(position_interpolator.utc(mean_time)))

                    sun_angle.append(det.sun_angle.value)
                    sun_positions.append(det.sun_position)
                    time.append(mean_time)
                    az, zen = det.earth_az_zen_sat
                    earth_az.append(az)
                    earth_zen.append(zen)
                    earth_position.append(det.earth_position)

                    quaternion.append(quaternion_step)
                    sc_pos.append(sc_pos_step)

                    p.increase()
        else:
            # Calculate the geometry for all times associated with this rank (for rank!=0).
            # No output here.
            for mean_time in list_times_to_calculate[times_lower_bound_index:times_upper_bound_index]:
                quaternion_step = position_interpolator.quaternion(mean_time)
                sc_pos_step = position_interpolator.sc_pos(mean_time)
                det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                   sc_pos=sc_pos_step,
                                                   time=astro_time.Time(position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                sun_positions.append(det.sun_position)
                time.append(mean_time)
                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                quaternion.append(quaternion_step)
                sc_pos.append(sc_pos_step)

        # make the list numpy arrays
        sun_angle = np.array(sun_angle)
        sun_positions = np.array(sun_positions)
        time = np.array(time)
        earth_az = np.array(earth_az)
        earth_zen = np.array(earth_zen)
        earth_position = np.array(earth_position)

        quaternion = np.array(quaternion)
        sc_pos = np.array(sc_pos)

        # gather all results in rank=0
        sun_angle_gather = comm.gather(sun_angle, root=0)
        time_gather = comm.gather(time, root=0)
        sun_positions_gather = comm.gather(sun_positions, root=0)
        earth_az_gather = comm.gather(earth_az, root=0)
        earth_zen_gather = comm.gather(earth_zen, root=0)
        earth_position_gather = comm.gather(earth_position, root=0)

        quaternion_gather = comm.gather(quaternion, root=0)
        sc_pos_gather = comm.gather(sc_pos, root=0)

        # make one list out of this
        if rank == 0:
            sun_angle_gather = np.concatenate(sun_angle_gather)
            time_gather = np.concatenate(time_gather)
            sun_positions_gather = np.concatenate(sun_positions_gather)
            earth_az_gather = np.concatenate(earth_az_gather)
            earth_zen_gather = np.concatenate(earth_zen_gather)
            earth_position_gather = np.concatenate(earth_position_gather)

            quaternion_gather = np.concatenate(quaternion_gather)
            sc_pos_gather = np.concatenate(sc_pos_gather)

        # broadcast the final arrays again to all ranks
        sun_angle = comm.bcast(sun_angle_gather, root=0)
        time = comm.bcast(time_gather, root=0)
        sun_positions = comm.bcast(sun_positions_gather, root=0)
        earth_az = comm.bcast(earth_az_gather, root=0)
        earth_zen = comm.bcast(earth_zen_gather, root=0)
        earth_position = comm.bcast(earth_position_gather, root=0)

        quaternion = comm.bcast(quaternion_gather, root=0)
        sc_pos = comm.bcast(sc_pos_gather, root=0)

        # Return everything

        return sun_angle, sun_positions, time, earth_az, earth_zen, earth_position, quaternion, sc_pos, times_lower_bound_index, \
               times_upper_bound_index

    def _one_day_setup_geometery_no_mpi(self, day_number):
        """
        Run the geometry precalculation with mpi support. Only use this funtion if you do not use MPI
        """
        assert not using_mpi, 'This function is only available if you are not using mpi!'

        # Create the PositionInterpolator object with the infos from the poshist file
        position_interpolator = PositionInterpolator(poshist=self._pos_hist[day_number])

        # Get the times for which the geometry should be calculated for this day (Build a mask that masks all time bins
        # outside the start and stop day of this time bin

        masksmaller = self._list_times_to_calculate >= self._day_start_times[day_number]
        masklarger = self._list_times_to_calculate <= self._day_stop_times[day_number]

        masktot = masksmaller * masklarger
        list_times_to_calculate = self._list_times_to_calculate[masktot]

        # Init all lists
        sun_angle = []
        sun_positions = []
        time = []
        earth_az = []  # azimuth angle of earth in sat. frame
        earth_zen = []  # zenith angle of earth in sat. frame
        earth_position = []  # earth pos in icrs frame (skycoord)

        # Additionally save the quaternion and the sc_pos of every time step. Needed for PS later.
        quaternion = []
        sc_pos = []

        # Give some output how much of the geometry is already calculated (progress_bar)
        with progress_bar(len(list_times_to_calculate), title='Calculating sun and earth position') as p:
            # Calculate the geometry for all times
            for mean_time in list_times_to_calculate:
                quaternion_step = position_interpolator.quaternion(mean_time)
                sc_pos_step = position_interpolator.sc_pos(mean_time)
                det = gbm_detector_list[self._det](quaternion=quaternion_step,
                                                   sc_pos=sc_pos_step,
                                                   time=astro_time.Time(position_interpolator.utc(mean_time)))

                sun_angle.append(det.sun_angle.value)
                sun_positions.append(det.sun_position)
                time.append(mean_time)
                az, zen = det.earth_az_zen_sat
                earth_az.append(az)
                earth_zen.append(zen)
                earth_position.append(det.earth_position)

                quaternion.append(quaternion_step)
                sc_pos.append(sc_pos_step)

                p.increase()

        # Make the list numpy arrays
        sun_angle = np.array(sun_angle)
        time = np.array(time)
        sun_positions = np.array(sun_positions)
        earth_az = np.array(earth_az)
        earth_zen = np.array(earth_zen)
        earth_position = np.array(earth_position)

        quaternion = np.array(quaternion)
        sc_pos = np.array(sc_pos)

        # Return everything

        return sun_angle, sun_positions, time, earth_az, earth_zen, earth_position, quaternion, sc_pos

    def _add_start_stop(self, timelist, start_add, stop_add):
        """
        Function that adds the times in start_add and stop_add to timelist if they are not already in the list
        :param timelist: list of times
        :param start_add: start of all days
        :param stop_add: stop of all days
        :return: timelist with start_add and stop_add times added
        """
        for start in start_add:
            if start not in timelist:
                timelist = np.append(timelist, start)
        for stop in stop_add:
            if stop not in timelist:
                timelist = np.append(timelist, stop)
        timelist.sort()
        return timelist
