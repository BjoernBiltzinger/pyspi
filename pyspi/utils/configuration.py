from omegaconf import OmegaConf
from dataclasses import dataclass
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

from pyspi.io.package_data import get_path_of_user_config


@dataclass(frozen=True)
class OnlineResources:
    catalog_url: str = "https://www.isdc.unige.ch/browse/w3query.pl"
    local_data: str = "/afs/ipp-garching.mpg.de/mpe/gamma/instruments/integral/data/revolutions"
    remote_data: str = "ftp://isdcarc.unige.ch/arc/rev_3/scw"


class DataAccess(IntEnum):
    AFS = "afs"
    ISDC = "isdc"


@dataclass
class Config:
    #    logging: Logging = Logging()
    resources: OnlineResources = OnlineResources()
    data_access: DataAccess = DataAccess.AFS
    internal_data_path: str = str(Path("~/spi_internal_data").expanduser())
    observation_data_path: str = str(Path("~/spi_data").expanduser())


# Read the default Config

pyspi_config: Config = OmegaConf.structured(Config)


# now glob the config directory

for user_config_file in get_path_of_user_config().glob("*.yml"):

    _partial_conf = OmegaConf.load(user_config_file)

    threeML_config: Config = OmegaConf.merge(pyspi_config, _partial_conf)
