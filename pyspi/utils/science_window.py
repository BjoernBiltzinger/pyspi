from collections import OrderedDict
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from astropy.time import Time

from pyspi.io.get_files import get_files


@dataclass
class ScienceWindow:

    id: str
    sw_type: str
    ra_x: float
    dec_x: float
    ra_z: float
    dec_z: float
    start_date: Time
    end_date: Time
    obs_type: str
    ps: str
    pi_name: str
    good_spi: int
    spi_mode: int

    def get(self) -> None:
        """
        Downloads the science window

        :returns:

        """

        get_files(self.id)


class ScienceWindowSet:
    def __init__(self, *science_windows: List[ScienceWindow]) -> None:

        # sort by time

        _idx = []

        for sw in science_windows:

            _idx.append(sw.start_time.mjd)

        idx = np.argsort(_idx)

        self._sw: Dict[str, ScienceWindow] = OrderedDict()

        self._n_windows: int = len(self._sw)

        for i in idx:

            self._sw[science_windows[i].id] = science_windows[i]

    def _to_subset(self, choice: np.ndarray) -> "ScienceWindowSet":
        """
        produces a filtered subset of the science windows
        """

        output = []

        for i, (k, v) in enumerate(self._sw.items()):

            if choice[i]:

                output.append(v)

        return ScienceWindowSet(*output)

    def get(self) -> None:

        for k, v in self._sw.items():

            v.get()

    def get_filtered_set(
        self,
        pointing_type: Optional[str] = "pointing",
        public: bool = True,
        obs_type: Optional[str] = "general",
    ) -> "ScienceWindowSet":

        idx = np.ones(self._n_windows, dtype=bool)

        if pointing_type is not None:

            for i, (k, v) in enumerate(self._sw.items()):

                if v.sw_type.lower() != pointing_type.lower():

                    idx[i] = False

        if obs_type is not None:

            for i, (k, v) in enumerate(self._sw.items()):

                if v.obs_type.lower() != obs_type.lower():

                    idx[i] = False

        for i, (k, v) in enumerate(self._sw.items()):

            if v.obs_type.lower() == "private":

                idx[i] = not public

            else:

                idx[i] = public

        return self._to_subset(idx)
