# Test spi pointing

def test_spipointing():
    from pyspi.utils.response.spi_pointing import SPIPointing
    from pyspi.io.package_data import get_path_of_external_data_dir
    from pyspi.io.get_files import get_files
    import os
    import numpy as np
    
    pointing_id = 169600130010
    get_files(pointing_id)

    geom_save_path = os.path.join(get_path_of_external_data_dir(),
                                  'pointing_data',
                                  str(pointing_id),
                                  'sc_orbit_param.fits.gz')

    point = SPIPointing(geom_save_path)

    assert np.sum(point.sc_matrix) == 1222.1378651210798, \
        'SPI pointing test failed'
