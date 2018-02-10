import numpy as np
import h5py


class SPIResponse(object):
    def __init__(self):

        self._load_irfs()

    def _load_irfs(self):

        irf_file = '/Users/jburgess/coding/pyspi/pyspi/data/spi_irfs.hdf5'

        irf_database = h5py.File(irf_file, 'r')

        self._energies = irf_database['energies'].value

        irf_data = irf_database['irfs']

        self._irfs = irf_data.value

        self._irf_xmin = irf_data.attrs['irf_xmin']
        self._irf_ymin = irf_data.attrs['irf_ymin']
        self._irf_xbin = irf_data.attrs['irf_xbin']
        self._irf_ybin = irf_data.attrs['irf_ybin']
        self._irf_nx = irf_data.attrs['nx']
        self._irf_ny = irf_data.attrs['ny']

        irf_database.close()

    def get_xy_pos(self, azimuth, zenith):

        x_pos = (zenith * np.cos(azimuth) - self._irf_xmin) / self._irf_xbin
        y_pos = (zenith * np.sin(azimuth) - self._irf_ymin) / self._irf_ybin

        return x_pos, y_pos

    def set_binned_data_energy_bounds(self, ebounds):

        self._ene_min = ebounds[:-1]
        self._ene_max = ebounds[1:]
        self._ebounds = ebounds

    def effective_area_per_detector(self, azimuth, zenith):

        x, y = self.get_xy_pos(azimuth, zenith)

        wgt, xx, yy = self._get_irf_weights(x, y)
        weighted_irf = spi.irfs[..., xx, yy].dot(wgt)

        return weighted_irf

    def _get_irf_weights(self, x_pos, y_pos):

        ix_left = x_pos if (x_pos >= 0.0) else x_pos - 1
        iy_low = y_pos if (y_pos >= 0.0) else y_pos - 1
        ix_right = ix_left + 1
        iy_up = iy_low + 1
        wgt_right = x_pos - ix_left
        wgt_up = y_pos - iy_low

        wgt = np.zeros(4)
        inx = np.zeros(4, dtype=int)

        if ix_left < 0.:

            if ix_right < 0.:

                left_low = [int(ix_left), int(iy_low)]
                right_low = [int(ix_right), int(iy_low)]
                left_up = [int(ix_left), int(iy_up)]
                right_up = [int(ix_right), int(iy_up)]

                out = np.array([left_low, right_low, left_up, right_up]).T

                return wgt, out[0], out[1]

            else:

                ix_left = ix_right
                wgt_left = 0.5
                wgt_right = 0.5

        elif ix_right >= self._irf_nx:

            if ix_left >= self._irf_nx:

                left_low = [int(ix_left), int(iy_low)]
                right_low = [int(ix_right), int(iy_low)]
                left_up = [int(ix_left), int(iy_up)]
                right_up = [int(ix_right), int(iy_up)]

                out = np.array([left_low, right_low, left_up, right_up]).T

                return wgt, out[0], out[1]

            else:

                ix_right = ix_left
                wgt_left = 0.5
                wgt_right = 0.5

        else:

            wgt_left = 1. - wgt_right

        if iy_low < 0:
            if iy_up < 0:

                left_low = [int(ix_left), int(iy_low)]
                right_low = [int(ix_right), int(iy_low)]
                left_up = [int(ix_left), int(iy_up)]
                right_up = [int(ix_right), int(iy_up)]

                out = np.array([left_low, right_low, left_up, right_up]).T

                return wgt, out[0], out[1]
            else:
                iy_low = iy_up
                wgt_up = 0.5
                wgt_low = 0.5

        elif iy_up >= self._irf_ny:

            if iy_low >= self._irf_ny:
                left_low = [int(ix_left), int(iy_low)]
                right_low = [int(ix_right), int(iy_low)]
                left_up = [int(ix_left), int(iy_up)]
                right_up = [int(ix_right), int(iy_up)]

                out = np.array([left_low, right_low, left_up, right_up]).T

                return wgt, out[0], out[1]

            else:

                iy_up = iy_low
                wgt_up = 0.5
                wgt_low = 0.5

        else:

            wgt_low = 1. - wgt_up

        left_low = [int(ix_left), int(iy_low)]
        right_low = [int(ix_right), int(iy_low)]
        left_up = [int(ix_left), int(iy_up)]
        right_up = [int(ix_right), int(iy_up)]

        wgt[0] = wgt_left * wgt_low
        wgt[1] = wgt_right * wgt_low
        wgt[2] = wgt_left * wgt_up
        wgt[3] = wgt_right * wgt_up

        out = np.array([left_low, right_low, left_up, right_up]).T

        return wgt, out[0], out[1]

    def get_irf_weights_vector(self, x_pos, y_pos):

        raise NotImplementedError('Cannot do this yet')

        idx_x_neg = x_pos < 0.
        idx_y_neg = y_pos < 0.

        ix_left = x_pos
        iy_low = y_pos

        ix_left[idx_x_neg] -= 1
        iy_low[idx_y_neg] -= 1

        ix_right = ix_left + 1
        iy_up = iy_low + 1
        wgt_right = x_pos - ix_left
        wgt_up = y_pos - iy_low

        wgt_left = 1. - wgt_right
        wgt_low = 1. - wgt_up

        ############################################

        selection = (ix_left < 0.) & (ix_right >= 0.)

        wgt_left[selection] = 0.5
        wgt_right[selection] = 0.5

        ix_left[selection] = ix_right[selection]

        selection = (ix_right >= self._irf_nx) & (ix_left < self._irf_nx)

        wgt_left[selection] = 0.5
        wgt_right[selection] = 0.5

        ix_right[selection] = ix_left[selection]

        selection = (iy_low < 0) & (iy_up >= 0)

        iy_low[selection] = iy_up[selection]
        wgt_up[selection] = 0.5
        wgt_low[selection] = 0.5

        selection = (iy_up >= self._irf_ny) & (iy_low < self._irf_ny)

        iy_up[selection] = iy_low[selection]
        wgt_up[selection] = 0.5
        wgt_low[selection] = 0.5

        #         inx[0] = int(ix_left + iy_low * self._irf_nx)
        #         inx[1] = int(ix_right + iy_low * self._irf_nx)
        #         inx[2] = int(ix_left + iy_up * self._irf_nx)
        #         inx[3] = int(ix_right + iy_up * self._irf_nx)

        left_low = [int(ix_left), int(iy_low)]
        right_low = [int(ix_right), int(iy_low)]
        left_up = [int(ix_left), int(iy_up)]
        right_up = [int(ix_right), int(iy_up)]

        wgt[0] = wgt_left * wgt_low
        wgt[1] = wgt_right * wgt_low
        wgt[2] = wgt_left * wgt_up
        wgt[3] = wgt_right * wgt_up

        return inx, wgt,

    @property
    def irfs(self):

        return self._irfs
