import os
from pathlib import Path

from ..utils import pyspi_config


def get_path_of_external_data_dir() -> Path:
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ.get("PYSPI")

    if file_path is None:

        file_path = pyspi_config.observation_data_path

    return Path(file_path)


def get_path_of_internal_data_dir() -> Path:
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ.get("PYSPI_PACKAGE_DATA")

    if file_path is None:

        file_path = pyspi_config.internal_data_path

    return Path(file_path)
