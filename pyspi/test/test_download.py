# Test download from isdarc for given pointin_id

def test_download():
    from pyspi.io.get_files import get_files_isdcarc
    from pyspi.io.package_data import get_path_of_external_data_dir
    from pyspi.io.file_utils import file_existing_and_readable
    import os

    pointing_id = 169600130010
    get_files_isdcarc(pointing_id)

    geom_save_path = os.path.join(get_path_of_external_data_dir(), 'pointing_data', str(pointing_id), 'sc_orbit_param.fits.gz')
    data_save_path = os.path.join(get_path_of_external_data_dir(), 'pointing_data', str(pointing_id), 'spi_oper.fits.gz')

    assert file_existing_and_readable(geom_save_path), 'Download test failed'
    assert file_existing_and_readable(data_save_path), 'Download test failed'


    
