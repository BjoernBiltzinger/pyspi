import pytest


def test_time_series_without_response():
    from pyspi.utils.data_builder.time_series_builder import \
        TimeSeriesBuilderSPI
    import numpy as np
    from astropy.time import Time
    ebounds = np.geomspace(20, 400, 30)
    grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
    tsb = TimeSeriesBuilderSPI.from_spi_grb("SPIDet0",
                                            0,
                                            grbtime,
                                            ebounds=ebounds,
                                            sgl_type="both",
                                            )
    assert np.isclose(tsb.time_series.
                      count_per_channel_over_interval(-500, 500)[10],
                      2120)


def test_time_series_with_response():
    from pyspi.utils.data_builder.time_series_builder import \
        TimeSeriesBuilderSPI
    import numpy as np
    from astropy.time import Time
    from pyspi.utils.function_utils import find_response_version
    from pyspi.utils.response.spi_response_data import ResponseDataRMF
    from pyspi.utils.response.spi_response import ResponseRMFGenerator
    from pyspi.utils.response.spi_drm import SPIDRM
    ebounds = np.geomspace(20, 400, 30)
    ein = np.geomspace(20, 800, 300)
    grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
    version = find_response_version(grbtime)
    rsp_base = ResponseDataRMF.from_version(version)
    ra_val = 94.6783
    dec_val = -70.99905
    drm_generator = ResponseRMFGenerator.from_time(grbtime,
                                                   0,
                                                   ebounds,
                                                   ein,
                                                   rsp_base)
    sd = SPIDRM(drm_generator, ra_val, dec_val)
    tsb = TimeSeriesBuilderSPI.from_spi_grb("SPIDet0",
                                            0,
                                            grbtime,
                                            response=sd,
                                            sgl_type="both",
                                            )

    assert np.isclose(tsb.time_series.
                      count_per_channel_over_interval(-500, 500)[10],
                      2120)

    tsb = TimeSeriesBuilderSPI.from_spi_grb("SPIDet0",
                                            0,
                                            grbtime,
                                            ebounds=ebounds,
                                            response=sd,
                                            sgl_type="both",
                                            )
    assert np.isclose(tsb.time_series.
                      count_per_channel_over_interval(-500, 500)[10],
                      2120)

    with pytest.raises(AssertionError):
        tsb = TimeSeriesBuilderSPI.from_spi_grb("SPIDet0",
                                                0,
                                                grbtime,
                                                ebounds=ein,
                                                response=sd,
                                                sgl_type="both",
                                                )
