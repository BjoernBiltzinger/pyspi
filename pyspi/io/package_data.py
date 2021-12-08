import os
from pathlib import Path


def get_path_of_external_data_dir() -> Path:
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ["PYSPI"]

    return Path(file_path)


def get_path_of_internal_data_dir() -> Path:
    """
    Get path to the external data directory (mostly to store data there)
    """
    file_path = os.environ["PYSPI_PACKAGE_DATA"]

    return Path(file_path)


def get_path_of_user_config() -> Path:

    if _custom_config_path is not None:

        config_path: Path = Path(_custom_config_path)

    config_path: Path = Path().home() / ".config" / "pyspi"

    if not config_path.exists():

        config_path.mkdir(parents=True)

    return config_path
