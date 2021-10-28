import os


def get_path_of_external_data_dir():
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ['PYSPI']

    return file_path


def get_path_of_internal_data_dir():
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ['PYSPI_PACKAGE_DATA']

    return file_path
