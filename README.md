[![CI tests](https://github.com/BjoernBiltzinger/pyspi/actions/workflows/publish_pypi.yml/badge.svg)](https://github.com/BjoernBiltzinger/pyspi/actions/workflows/publish_pypi.yml)
[![Docs](https://github.com/BjoernBiltzinger/pyspi/actions/workflows/docs.yml/badge.svg)](https://pyspi.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/BjoernBiltzinger/pyspi/branch/master/graph/badge.svg)](https://codecov.io/gh/BjoernBiltzinger/pyspi)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6376003.svg)](https://doi.org/10.5281/zenodo.6376003)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.04017/status.svg)](https://doi.org/10.21105/joss.04017)
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

# Contributing 

Contributions to ```PySPI``` are always welcome. They can come in the form of:

## Issues

Please use the [Github issue tracking system for any bugs](https://github.com/BjoernBiltzinger/pyspi/issues), for questions, bug reports and or feature requests.

## Add to Source Code

To directly contribute to the source code of ```PySPI```, please fork the Github repository, add the changes to one of the branches in your forked repository and then create a [pull request to the master of the main repository](https://github.com/BjoernBiltzinger/pyspi/pulls) from this branch. Code contribution is welcome for different topics:

### Add Functionality

If ```PySPI``` is missing some functionality that you need, you can either create an issue in the Github repository or add it to the code and create a pull request. Always make sure that the old tests do not break and adjust them if needed. Also please add tests and documentation for the new functionality in the pyspi/test folder. This ensures that the functionality will not get broken by future changes to the code and other people will know that this feature exists.

### Code Improvement

You can also contribute code improvements, like making calculations faster or improve the style of the code. Please make sure that the results of the software do not change in this case.

### Bug Fixes

Fixing bugs that you found or that are mentioned in one of the issues is also a good way to contribute to ```PySPI```. Please also make sure to add tests for your changes to check that the bug is gone and that the bug will not recur in future versions of the code.

### Documentation

Additions or examples, tutorials, or better explanations are always welcome. To ensure that the documentation builds with the current version of the software, we are using [jupytext](https://jupytext.readthedocs.io/en/latest/) to write the documentation in Markdown. These are automatically converted to and executed as jupyter notebooks when changes are pushed to Github. 

## Testing

If one wants to run the test suite, simply install `pytest` and `pytest-cov`, then run

```bash
pytest -v

```

in the top level directory. 


