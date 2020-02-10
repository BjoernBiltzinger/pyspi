# Test package data

def test_packagedata():
    from pyspi.io.package_data import get_path_of_data_file, get_path_of_data_dir, get_path_of_external_data_dir
    
    path = get_path_of_data_file('testtest.test')
    pyspi_folder = get_path_of_data_dir()
    assert path==pyspi_folder+'/testtest.test', 'Package data test failed'

    get_path_of_external_data_dir()
