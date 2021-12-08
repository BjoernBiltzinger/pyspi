import collections
from pathlib import Path
from collections import OrderedDict, MutableMapping
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import numpy as np
from astropy.time import Time

from pyspi.io.get_files import get_files
from pyspi.io.package_data import get_path_of_external_data_dir


@dataclass
class ObservationFiles:
    orbit: Path
    house_keeping: Path
    observation: Path


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
        Downloads the science window data

        :returns:

        """

        get_files(self.id)

    @property
    def files(self) -> ObservationFiles:

        """
        The files associated with these data

        :returns:

        """
        base_path: Path = (
            get_path_of_external_data_dir() / "pointing_data" / self.id
        )

        return ObservationFiles(
            orbit=base_path / "sc_orbit_param.fits.gz",
            house_keeping=base_path / "spi_science_hk.fits.gz",
            observation=base_path / "spi_oper.fits.gz",
        )


class ScienceWindowSet(MutableMapping):
    def __init__(self, *science_windows: List[ScienceWindow]) -> None:

        # sort by time

        _idx = []

        for sw in science_windows:

            _idx.append(sw.start_date.mjd)

        idx = np.argsort(_idx)

        sw: Dict[str, ScienceWindow] = OrderedDict()

        # self._n_windows: int = len(science_windows)

        for i in idx:

            sw[science_windows[i].id] = science_windows[i]

        self.__dict__.update(**sw)

    def _to_subset(self, choice: np.ndarray) -> "ScienceWindowSet":
        """
        produces a filtered subset of the science windows
        """

        output = []

        for i, (k, v) in enumerate(self.items()):

            if choice[i]:

                output.append(v)

        return ScienceWindowSet(*output)

    def get(self) -> None:

        """
        Download all science windows

        :returns:

        """
        for k, v in self.items():

            v.get()

    def get_filtered_set(
        self,
        pointing_type: Optional[str] = "pointing",
        public: bool = True,
        obs_type: Optional[str] = "general",
    ) -> "ScienceWindowSet":
        """
        down select the data

        :param pointing_type:
        :param public:
        :param obs_type:
        """

        idx = np.ones(len(self), dtype=bool)

        if pointing_type is not None:

            for i, (k, v) in enumerate(self.items()):

                if v.sw_type.lower() != pointing_type.lower():

                    idx[i] = False

        if obs_type is not None:

            for i, (k, v) in enumerate(self.items()):

                if v.obs_type.lower() != obs_type.lower():

                    idx[i] = False

        for i, (k, v) in enumerate(self.items()):

            if v.ps.lower() == "private":

                idx[i] = not public

            else:

                idx[i] = public

        return self._to_subset(idx)

    @property
    def files(self) -> Dict[str, ObservationFiles]:
        """
        The files associated with the science windows
        """

        out = collections.OrderedDict()

        for k, v in self.items():

            out[k] = v.files

        return out

    def __setitem__(self, key, value):

        raise RuntimeError()

        self.__dict__[key] = value

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        raise RuntimeError()

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    # The final two methods aren't required, but nice for demo purposes:
    def __str__(self):
        """returns simple dict representation of the mapping"""
        return str(self.__dict__)

    def __repr__(self):
        """echoes class, id, & reproducible representation in the REPL"""
        return "{}, {}".format(super().__repr__(), self.__dict__)
