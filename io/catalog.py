from typing import Optional, List, Dict
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

    def _prepare_payload(self, **kwargs) -> Dict[str, Any]:

        initial_inputs = dict(
            mission="integral_rev3_scw", sortvar="START_DATE", fields="All"
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
        query=None,
    ):

        if coord is not None:

            user_coord: SkyCoord = coord

        else:

            user_coord = SkyCoord(ra=ra, dec=dec, unit="deg", frame="icrs")

        inputs = self._prepare_payload(position=user_coord, radius=radius)

        initial_table = self._heasarc.query_region(inputs)

        table = self._process_result(initial_table)

        sws: List[ScienceWindow] = []

        for k, v in table.iterrows():

            sws.append(
                ScienceWindow(
                    id=k,
                    sw_type=v.SW_TYPE,
                    ra_x=v.RA_X,
                    dec_x=v.DEC_X,
                    ra_z=v.RA_Z,
                    dec_z=v.DEC_Z,
                    start_date=_format_time(v.START_DATE),
                    end_date=_format_time(v.END_DATE),
                    obs_type=v.OBS_TYPE,
                    ps=v.PS,
                    pi_name=v.PI_NAME,
                    good_spi=v.GOOD_SPI,
                    spi_mode=v.SPI_MODE,
                )
            )

        self._last_table = table

        self._science_windows = ScienceWindowSet(*sws)

    def query_object(
        self,
        object_name: str,
        radius: u.Unit = 1 * u.deg,
        query=None,
    ):

        inputs = self._prepare_payload(object_name=object_name, radius=radius)

        initial_table = self._heasarc.query_region(inputs)

        table = self._process_result(initial_table)

        sws: List[ScienceWindow] = []

        for k, v in table.iterrows():

            sws.append(
                ScienceWindow(
                    id=k,
                    sw_type=v.SW_TYPE,
                    ra_x=v.RA_X,
                    dec_x=v.DEC_X,
                    ra_z=v.RA_Z,
                    dec_z=v.DEC_Z,
                    start_date=_format_time(v.START_DATE),
                    end_date=_format_time(v.END_DATE),
                    obs_type=v.OBS_TYPE,
                    ps=v.PS,
                    pi_name=v.PI_NAME,
                    good_spi=v.GOOD_SPI,
                    spi_mode=v.SPI_MODE,
                )
            )

        self._last_table = table

        self._science_windows = ScienceWindowSet(*sws)
