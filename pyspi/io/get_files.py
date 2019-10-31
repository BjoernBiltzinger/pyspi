import os
from shutil import copyfile
from pyspi.io.package_data import get_path_of_external_data_dir
from astropy.utils.data import download_file

from pyspi.io.file_utils import file_existing_and_readable

def get_files_afs(pointing_id):
    """
    Function to copy the needed files for a certain pointing_id to the local file system.
    :param pointing_id: Id of pointing e.g. '180100610010' as string!
    :return:
    """
    
    # If pointing_id is given as integer, convert it to string
    pointing_id = str(pointing_id)

    # Path to pointing_id directory
    dir_path = '/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/data/revolutions/{}/{}.001/'.format(pointing_id[:4], pointing_id)
    
    # Paths to the data file and the orbit file on the afs server
    geom_afs_path = os.path.join(dir_path, 'sc_orbit_param.fits.gz') 
    data_afs_path = os.path.join(dir_path, 'spi_oper.fits.gz')

    # Path where data should be stored
    geom_save_path = os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'sc_orbit_param.fits.gz')
    data_save_path = os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id, 'spi_oper.fits.gz')

    # Check if file structure exists. If not, create it.
    if not os.path.exists(os.path.join(get_path_of_external_data_dir())):
        os.mkdir(os.path.join(get_path_of_external_data_dir()))
    if not os.path.exists(os.path.join(get_path_of_external_data_dir(), 'pointing_data')):
        os.mkdir(os.path.join(get_path_of_external_data_dir(), 'pointing_data'))
    if not os.path.exists(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id)):
        os.mkdir(os.path.join(get_path_of_external_data_dir(), 'pointing_data', pointing_id))

    # If needed datafiles are not already in the right path copy them there
    if not file_existing_and_readable(geom_save_path):
        assert os.path.exists(dir_path), 'Either pointin_id is not valid, or you have no access to the afs server or no rights to read the integral data' 
        copyfile(geom_afs_path, geom_save_path)

    if not file_existing_and_readable(data_save_path):
        assert os.path.exists(dir_path), 'Either pointin_id is not valid, or you have no access to the afs server or no rights to read the integral data' 
        copyfile(data_afs_path, data_save_path)
        
