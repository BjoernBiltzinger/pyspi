import os
from shutil import copyfile
from pyspi.io.package_data import get_path_of_external_data_dir
from astropy.utils.data import download_file
import requests
import shutil
import urllib

from pyspi.io.file_utils import file_existing_and_readable

def create_file_structure(pointing_id):
    """
    Create the file structure to save the datafiles
    :param pointing_id: Id of pointing e.g. '180100610010' as string!
    :return:
    """
    # Check if file structure exists. If not, create it.
    if not os.path.exists(get_path_of_external_data_dir()):
        os.mkdir(get_path_of_external_data_dir())

    if not os.path.exists(os.path.join(get_path_of_external_data_dir(),
                                       'pointing_data')):
        os.mkdir(os.path.join(get_path_of_external_data_dir(),
                              'pointing_data'))

    if not os.path.exists(os.path.join(get_path_of_external_data_dir(),
                                       'pointing_data',
                                       pointing_id)):
        os.mkdir(os.path.join(get_path_of_external_data_dir(),
                              'pointing_data',
                              pointing_id))


def get_and_save_file(file_path, file_save_path, access="isdc"):
    """
    Function to get and save a file located at file_path to file_save_path
    :param file_path: File location (link or path to afs)
    :param file_save_path: File Save location (on local system)
    :param access: How to get the data. Possible are "isdc" and "afs"
    :return:
    """
    assert access in ["isdc", "afs"],\
        f"Access variable must be 'isdc' or 'afs' but is {access}."

    if not file_existing_and_readable(file_save_path):
        if access == "afs":
            assert os.path.exists(file_path), "Either pointing_id "\
                "is not valid, or you have no access to the afs server "\
                "or no rights to read the integral data"

            copyfile(file_path, file_save_path)

        else:

            try:
                urllib.request.urlopen(file_path)

            except:

                raise AssertionError(f'Link {file_path} does not exists!')

            data = download_file(file_path)
            shutil.move(data, file_save_path)


def get_files(pointing_id, access="isdc"):
    """
    Function to get the needed files for a certain pointing_id and save
    them in the correct folders.
    :param pointing_id: Id of pointing e.g. '180100610010' as string or int
    :param access: How to get the data. Possible are "isdc" and "afs"
    :return:
    """
    # If pointing_id is given as integer, convert it to string
    pointing_id = str(pointing_id)

    assert access in ["isdc", "afs"],\
        f"Access variable must be 'isdc' or 'afs' but is {access}."

    # Path where data should be stored
    geom_save_path = os.path.join(get_path_of_external_data_dir(),
                                  'pointing_data',
                                  pointing_id,
                                  'sc_orbit_param.fits.gz')
    data_save_path = os.path.join(get_path_of_external_data_dir(),
                                  'pointing_data',
                                  pointing_id,
                                  'spi_oper.fits.gz')
    hk_save_path = os.path.join(get_path_of_external_data_dir(),
                                'pointing_data',
                                pointing_id,
                                'spi_science_hk.fits.gz')

    if access == "afs":
        # Path to pointing_id directory
        dir_link = "/afs/ipp-garching.mpg.de/mpe/gamma/"\
            "instruments/integral/data/revolutions/"\
            "{}/{}.001/".format(pointing_id[:4], pointing_id)

    else:

        dir_link = "ftp://isdcarc.unige.ch/arc/rev_3/scw/"\
            "{}/{}.001/".format(pointing_id[:4], pointing_id)

    # Paths to the data file and the orbit file on the afs server
    geom_path = os.path.join(dir_link,
                                 'sc_orbit_param.fits.gz')
    data_path = os.path.join(dir_link,
                                 'spi_oper.fits.gz')
    hk_path = os.path.join(dir_link,
                                'spi_science_hk.fits.gz')

    create_file_structure(pointing_id)

    # Get the data files we need
    get_and_save_file(geom_path, geom_save_path, access=access)
    get_and_save_file(data_path, data_save_path, access=access)
    get_and_save_file(hk_path, hk_save_path, access=access)
