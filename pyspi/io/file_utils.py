from pathlib import Path

from threeML.io.logging import setup_logger


log = setup_logger(__name__)


def sanitize_filename(filename, abspath: bool = False) -> Path:

    path: Path = Path(filename)

    sanitized = path.expanduser()

    if abspath:

        return sanitized.absolute()

    else:

        return sanitized


def file_existing_and_readable(filename) -> bool:

    sanitized_filename: Path = sanitize_filename(filename)

    return sanitized_filename.is_file()


def path_exists_and_is_directory(path) -> bool:

    sanitized_path: Path = sanitize_filename(path, abspath=True)

    return sanitized_path.is_dir()


def if_directory_not_existing_then_make(directory) -> None:
    """
    If the given directory does not exists, then make it

    :param directory: directory to check or make
    :return: None
    """

    sanitized_directory: Path = sanitize_filename(directory)

    try:

        sanitized_directory.mkdir(parents=True, exist_ok=False)

    except (FileExistsError):

        # should add logging here!

        pass
