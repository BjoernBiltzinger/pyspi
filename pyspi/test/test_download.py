# Test download from isdarc for given pointin_id
import pytest
import urllib.request
from urllib.error import URLError
try:
    urllib.request.urlopen("ftp://isdcarc.unige.ch/arc/rev_3/scw/"
                           "1189/118900580010.001/sc_orbit_param.fits.gz")
    run_test = False
except URLError:
    run_test = True


@pytest.mark.skipif(run_test, "ISDC data arciv is broken")
def test_download():
    from pyspi.io.get_files import get_files
    from pyspi.io.package_data import get_path_of_external_data_dir
    from pyspi.io.file_utils import file_existing_and_readable
    import os

    pointing_id = 169600130010
    base_path = os.path.join(get_path_of_external_data_dir(),
                             'pointing_data',
                             str(pointing_id))

    if os.path.exists(base_path):
        os.rmdir(base_path)

    get_files(pointing_id)

    geom_save_path = os.path.join(get_path_of_external_data_dir(),
                                  'pointing_data',
                                  str(pointing_id),
                                  'sc_orbit_param.fits.gz')
    data_save_path = os.path.join(get_path_of_external_data_dir(),
                                  'pointing_data',
                                  str(pointing_id),
                                  'spi_oper.fits.gz')

    assert file_existing_and_readable(geom_save_path), 'Download test failed'
    assert file_existing_and_readable(data_save_path), 'Download test failed'
