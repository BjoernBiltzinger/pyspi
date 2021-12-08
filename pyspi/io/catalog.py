from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd

from astroquery.heasarc import Heasarc, Conf
from astropy.coordinates import SkyCoord
from astropy.time import Time
import astropy.units as u

from pyspi.utils.science_window import ScienceWindow, ScienceWindowSet


def _format_time(time: str) -> Time:

    return Time("T".join(time.split()), format="isot", scale="utc")


class SPICatalog:
    def __init__(self):
        """
        Interface to the ISDC INTEGRAL Catalog. Can be used to
        obtain and build observations

        :returns:

        """

        # ISDC does not provide a VO service
        # so we will query the old school way

        Conf.server.set("https://www.isdc.unige.ch/browse/w3query.pl")

        self._heasarc = Heasarc()

        self._science_windows: Optional[ScienceWindowSet] = None

    def _prepare_payload(
        self, start_date, end_date, **kwargs
    ) -> Dict[str, Any]:

        if (start_date is None) and (end_date is None):

            time = None

        else:

            initital_time = " .. "

            if start_date is not None:

                initital_time = f"{start_date}{initital_time}"

            if end_date is not None:

                initital_time = f"{initital_time}{end_date}"

            time = initital_time

        initial_inputs = dict(
            mission="integral_rev3_scw",
            sortvar="START_DATE",
            fields="All",
            time=time,
        )

        for k, v in kwargs.items():

            initial_inputs[k] = v

        return initial_inputs

    def _process_result(self, initial_table, query: str) -> pd.DataFrame:

        initial_table.convert_bytestring_to_unicode()

        table = (
            initial_table.to_pandas()
            .set_index("SCW_ID")
            .sort_values("_SEARCH_OFFSET")
        )

        if query is not None:

            out = table.query(query)

        else:

            out = table

        return out

    def query_region(
        self,
        coord: Optional[SkyCoord] = None,
        ra: Optional[float] = None,
        dec: Optional[float] = None,
        radius: u.Unit = 1 * u.deg,
        start_date: Optional[str] = "2005-01-01",
        end_date: Optional[str] = None,
        query=None,
    ):

        if coord is not None:

            user_coord: SkyCoord = coord

        else:

            user_coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        inputs = self._prepare_payload(
            start_date, end_date, position=user_coord, radius=radius
        )

        initial_table = self._heasarc.query_region(**inputs)

        table = self._process_result(initial_table, query)

        sws: List[ScienceWindow] = []

        for k, v in table.iterrows():

            tmp = v.GOOD_SPI.strip()

            try:

                good_spi = int(tmp)

            except ValueError:

                good_spi = -99

            sws.append(
                ScienceWindow(
                    id=k,
                    sw_type=v.SCW_TYPE.strip(),
                    ra_x=float(v.RA_X),
                    dec_x=float(v.DEC_X),
                    ra_z=float(v.RA_Z),
                    dec_z=float(v.DEC_Z),
                    start_date=_format_time(v.START_DATE),
                    end_date=_format_time(v.END_DATE),
                    obs_type=v.OBS_TYPE.strip(),
                    ps=v.PS.strip(),
                    pi_name=v.PI_NAME.strip().replace(r"\n", " "),
                    good_spi=good_spi,
                    spi_mode=int(v.SPIMODE.strip()),
                )
            )

        self._last_table = table

        self._science_windows = ScienceWindowSet(*sws)

    def query_object(
        self,
        object_name: str,
        radius: u.Unit = 1 * u.deg,
        start_date: Optional[str] = "2005-01-01",
        end_date: Optional[str] = None,
        query=None,
    ):

        inputs = self._prepare_payload(
            start_date, end_date, object_name=object_name, radius=radius
        )

        initial_table = self._heasarc.query_object(**inputs)

        table = self._process_result(initial_table, query)

        sws: List[ScienceWindow] = []

        for k, v in table.iterrows():

            tmp = v.GOOD_SPI.strip()

            try:

                good_spi = int(tmp)

            except ValueError:

                good_spi = -99

            sws.append(
                ScienceWindow(
                    id=k,
                    sw_type=v.SCW_TYPE.strip(),
                    ra_x=float(v.RA_X),
                    dec_x=float(v.DEC_X),
                    ra_z=float(v.RA_Z),
                    dec_z=float(v.DEC_Z),
                    start_date=_format_time(v.START_DATE),
                    end_date=_format_time(v.END_DATE),
                    obs_type=v.OBS_TYPE.strip(),
                    ps=v.PS.strip(),
                    pi_name=v.PI_NAME.strip().replace(r"\n", " "),
                    good_spi=good_spi,
                    spi_mode=int(v.SPIMODE.strip()),
                )
            )

        self._last_table = table

        self._science_windows = ScienceWindowSet(*sws)

    @property
    def science_windows(self) -> ScienceWindowSet:

        return self._science_windows

    @property
    def current_table(self) -> pd.DataFrame:
        return self._last_table
