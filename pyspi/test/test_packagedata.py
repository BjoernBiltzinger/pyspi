# Test package data

def test_packagedata():
    from pyspi.io.package_data import (get_path_of_internal_data_dir,
                                       get_path_of_external_data_dir)
    

    get_path_of_internal_data_dir()
    get_path_of_external_data_dir()
