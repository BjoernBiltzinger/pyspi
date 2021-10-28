import numpy as np


def test_active_dets_and_response_version():
    from pyspi.utils.livedets import get_live_dets
    from pyspi.utils.function_utils import find_response_version

    time_string = "051212 205010"
    ld = get_live_dets(time_string, event_types="single")
    version = find_response_version(time_string)

    assert np.sum(ld) == 152, f"Got wrong active det numbers for {time_string}"
    assert version == 2, f"Got wrong rsp version number for {time_string}"

    time_string = "031212 205010"
    ld = get_live_dets(time_string, event_types="single")
    version = find_response_version(time_string)

    assert np.sum(ld) == 169, f"Got wrong active det numbers for {time_string}"
    assert version == 1, f"Got wrong rsp version number for {time_string}"

    time_string = "151212 205010"
    ld = get_live_dets(time_string, event_types="single")
    version = find_response_version(time_string)

    assert np.sum(ld) == 146, f"Got wrong active det numbers for {time_string}"
    assert version == 4, f"Got wrong rsp version number for {time_string}"


def test_plotting():
    from pyspi.io.plotting.spi_display import SPI
    time_string = "151212 205010"
    s = SPI(time=time_string)
    fig = s.plot_spi_working_dets()
