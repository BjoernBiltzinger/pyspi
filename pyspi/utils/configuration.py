from omegaconf import OmegaConf
from dataclasses import dataclass
from dataclasses import dataclass
from enum import IntEnum, Enum
from pathlib import Path


def get_path_of_user_config() -> Path:

    # if _custom_config_path is not None:

    #     config_path: Path = Path(_custom_config_path)

    config_path: Path = Path().home() / ".config" / "pyspi"

    if not config_path.exists():

        config_path.mkdir(parents=True)

    return config_path


@dataclass(frozen=True)
class OnlineResources:
    catalog_url: str = "https://www.isdc.unige.ch/browse/w3query.pl"
    local_data: str = "/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/data/revolutions"
    remote_data: str = "ftp://isdcarc.unige.ch/arc/rev_3/scw"


class DataAccess(Enum):
    AFS = "afs"
    ISDC = "isdc"


@dataclass
class Config:
    #    logging: Logging = Logging()
    resources: OnlineResources = OnlineResources()
    data_access: DataAccess = DataAccess.ISDC
    internal_data_path: str = str(Path("~/spi_internal_data").expanduser())
    observation_data_path: str = str(Path("~/spi_data").expanduser())


# Read the default Config

pyspi_config: Config = OmegaConf.structured(Config)


# now glob the config directory

for user_config_file in get_path_of_user_config().glob("*.yml"):

    _partial_conf = OmegaConf.load(user_config_file)

    threeML_config: Config = OmegaConf.merge(pyspi_config, _partial_conf)
