import numpy as np


def test_response():
    from astropy.time import Time
    from pyspi.utils.function_utils import find_response_version
    from pyspi.utils.response.spi_response_data import ResponseDataRMF
    from pyspi.utils.response.spi_response import ResponseRMFGenerator
    from pyspi.utils.response.spi_drm import SPIDRM

    grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
    ein = np.geomspace(20,8000,300)
    ebounds = np.geomspace(20,4000,30)

    version = find_response_version(grbtime)
    rsp_base = ResponseDataRMF.from_version(version)

    # Check the rsp_base values
    assert np.isclose(rsp_base.irfs_nonphoto_1.sum(), 1720731.0176313636), \
        "The IRF values in the rsp base object are not the ones we expect."

    assert np.isclose(rsp_base.ebounds_rmf_3_base.sum(), 74573.8), \
        "The Ebounds in the rsp base object are not the ones we expect."

    det = 0
    ra_val = 94.6783
    dec_val = -70.99905
    drm_generator = ResponseRMFGenerator.from_time(grbtime,
                                                   det,
                                                   ebounds,
                                                   ein,
                                                   rsp_base)
    sd = SPIDRM(drm_generator, ra_val, dec_val)

    assert sd.matrix.sum() == 1846.700886770058, \
        "We expect a different response matrix..."
