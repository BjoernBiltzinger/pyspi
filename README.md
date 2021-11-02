[![CI tests](https://github.com/BjoernBiltzinger/pyspi/actions/workflows/publish_pypi.yml/badge.svg)](https://github.com/BjoernBiltzinger/pyspi/actions/workflows/publish_pypi.yml)
[![Docs](https://github.com/BjoernBiltzinger/pyspi/actions/workflows/docs.yml/badge.svg)](https://pyspi.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/BjoernBiltzinger/pyspi/branch/master/graph/badge.svg)](https://codecov.io/gh/BjoernBiltzinger/pyspi)
# pyspi
![alt text](https://raw.githubusercontent.com/BjoernBiltzinger/pyspi/master/docs/media/pypsi_logo2.png)

A python analysis framework for INTEGRAL/SPI

```PySPI``` provides a plugin for [3ML](https://threeml.readthedocs.io/en/stable/) for INTEGRAL/SPI data, which allows to analyze GRB data at the moment. In the future we plan to also add support for non transient sources.

## Installation

### Pip
```PySPI``` can be installed via pip.
```bash
pip install py-spi
```

### Github

To install the latest release from Github run
```bash
git clone https://github.com/BjoernBiltzinger/pyspi.git
```
After that first install the packages from the requirement.txt file with
```bash
cd pyspi
pip install -r requirements.txt
```
Now you can install ```PySPI``` with
```bash
python setup.py install
```

### Additional Data Files

There are a few large data files for the background model and the response that are not included in the Github repository. To get these data files run and specify the path where this data folder should be stored on your local machine. Here you have to change the /path/to/internal/data with the path you want to use on your local computer.
```bash
wget https://grb.mpe.mpg.de/pyspi_datafolder && unzip pyspi_datafolder
mv data /path/to/internal/data && rm -f pyspi_datafolder
```

### Environment Variables

Next you have to set two environment variable. One to define the path to the folder of the external data like the different SPI data files that will be downloaded by ```PySPI``` and one to define the path to the internal data folder we downloaded earlier.
```bash
export PYSPI=/path/to/external/datafolder
export PYSPI_PACKAGE_DATA=/path/to/internal/data
```

You should add these two line to your bashrc (or similar) file to automatically set this variable in every new terminal.

Now we are ready to go.

## Features

Please have a look at the [documentation](https://pyspi.readthedocs.io/en/latest/) to check out the features ```PySPI``` provides. There is also a [full example](https://pyspi.readthedocs.io/en/latest/notebooks/grb_analysis/), how to perform a spectral fit for the data for GRB120711A, as well as how to localize the GRB with ```PySPI```.

## Contributing

Everyone is invited to contribute to ```PySPI```. If you want to contribute, please use the standard Github features, like opening issues and pull requests. Please also always make sure, that you add tests and documentation for the changes you want to include into the package.
