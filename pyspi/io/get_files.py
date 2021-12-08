import os
from shutil import copyfile
from pathlib import Path
from pyspi.io.package_data import get_path_of_external_data_dir
from astropy.utils.data import download_file
import requests
import shutil
import urllib
from urllib.error import URLError

from pyspi.io.file_utils import (
    file_existing_and_readable,
    if_directory_not_existing_then_make,
)

from ..utils import pyspi_config
from ..utils.configuration import DataAccess

def create_file_structure(pointing_id: str) -> None:
    """
    Create the file structure to save the datafiles
    :param pointing_id: Id of pointing e.g. '180100610010' as string!
    :return:
    """
    # Check if file structure exists. If not, create it.

    if_directory_not_existing_then_make(get_path_of_external_data_dir())

    if not os.path.exists(
        os.path.join(get_path_of_external_data_dir(), "pointing_data")
    ):
        os.mkdir(os.path.join(get_path_of_external_data_dir(), "pointing_data"))

    if not os.path.exists(
        os.path.join(
            get_path_of_external_data_dir(), "pointing_data", pointing_id
        )
    ):
        os.mkdir(
            os.path.join(
                get_path_of_external_data_dir(), "pointing_data", pointing_id
            )
        )



def get_and_save_file(
    extension: str,
    pointing_id: str,
    access: str = pyspi_config.data_access.value,
):
    """
    Function to get and save a file located at file_path to file_save_path
    :param extension: File name you want to download
    :param pointing_id: The id of the pointing
    :param access: How to get the data. Possible are "isdc" and "afs"
    :return:
    """
    assert access.lower() in [
        "isdc",
        "afs",
    ], f"Access variable must be 'isdc' or 'afs' but is {access}."

    save_path: Path = get_path_of_external_data_dir() / "pointing_data" / pointing_id / extension
    )

    _base_path: Path =  Path(pointing_id[:4]) / f"{pointing_id}.001" / extension

    if not file_existing_and_readable(save_path):
        if access == "afs":
            # Path to pointing_id directory
            file_path = Path(pyspi_config.resources.local_data) / _base_path

            assert os.path.exists(file_path), (
                "Either pointing_id "
                "is not valid, or you have no access to the afs server "
                "or no rights to read the integral data"
            )

            copyfile(file_path, save_path)

        else:
            # use ISDC ftp server
            try:

                file_path = Path(pyspi_config.resources.remote_data) / _base_path
                
                urllib.request.urlopen(file_path)
                data = download_file(file_path)
                shutil.move(data, save_path)
            except URLError:
                # try rsync
                file_path = Path("isdcarc.unige.ch::arc") / "rev_3"/ "scw" / _base_path
    

                os.system(f"rsync -lrtv {file_path} {save_path}")

            except Exception as e:
                raise AssertionError(
                    f"Downloading {file_path} from the ISDC"
                    f"does not work! Error: {e}"
                )


def get_files(pointing_id: str, access: str =  pyspi_config.data_access.value) -> None:
    """
    Function to get the needed files for a certain pointing_id and save
    them in the correct folders.
    :param pointing_id: Id of pointing e.g. '180100610010' as string or int
    :param access: How to get the data. Possible are "isdc" and "afs"
    :return:
    """
    # If pointing_id is given as integer, convert it to string
    pointing_id = str(pointing_id)

    assert access in [
        "isdc",
        "afs",
    ], f"Access variable must be 'isdc' or 'afs' but is {access}."

    create_file_structure(pointing_id)

    # Get the data files we need
    get_and_save_file(
        extension="sc_orbit_param.fits.gz",
        pointing_id=pointing_id,
        access=access,
    )
    get_and_save_file(
        extension="spi_oper.fits.gz", pointing_id=pointing_id, access=access
    )
    get_and_save_file(
        extension="spi_science_hk.fits.gz",
        pointing_id=pointing_id,
        access=access,
    )
