import pkg_resources
import os


def get_path_of_data_file(data_file):
    """
    Get path to data file in the intern data directory
    :param data_file: Name of data file
    :return: Path to wanted data file in the intern data directory
    """
    file_path = pkg_resources.resource_filename("pyspi", 'data/%s' % data_file)

    return file_path


def get_path_of_data_dir():
    """
    Get path to intern data directory
    :return: Path to intern data directory
    """
    file_path = pkg_resources.resource_filename("pyspi", 'data')

    return file_path


def get_path_of_external_data_dir():
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ['PYSPI']

    return file_path
