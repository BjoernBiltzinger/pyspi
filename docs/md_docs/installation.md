---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.7.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->

# Installation

## Pip
To install PySPI via pip just use
```bash
pip install py-spi
```

## Conda/Mamba

If you have problems installing PySPI within a Conda environment try to create your environment with this command

```bash
conda create -n pyspi -c conda-forge python=3.9 numpy scipy ipython numba astropy matplotlib h5py pandas pytables
```

or for Mamba

```bash
mamba create -n pyspi -c conda-forge python=3.9 numpy scipy ipython numba astropy matplotlib h5py pandas pytables
```

and then run 

```bash
pip install py-spi
```

with the environment activated.

## Github

To install the latest release from Github run
```bash
git clone https://github.com/BjoernBiltzinger/pyspi.git
```
After that first install the packages from the requirement.txt file with
```bash
cd pyspi
pip install -r requirements.txt
```
Now you can install PySPI with
```bash
python setup.py install
```

## Additional Data Files

There are a few large data files for the background model and the response that are not included in the Github repository. To get these data files run the following commands. Here the data folder is downloaded and is moved to a user defined path where this data folder should be stored on your local machine. Here you have to change the /path/to/internal/data to the path you want to use on your local computer. This only needs to be downloaded once and will not change afterwards.
```bash
wget https://grb.mpe.mpg.de/pyspi_datafolder && unzip pyspi_datafolder
mv data /path/to/internal/data && rm -f pyspi_datafolder
```

## Environment Variables

Next you have to set two environment variable. One to define the path to the folder of the external data like the different SPI data files that will be downloaded by PySPI and one to define the path to the internal data folder we downloaded earlier.
```bash
export PYSPI=/path/to/external/datafolder
export PYSPI_PACKAGE_DATA=/path/to/internal/data
```
Here /path/to/external/datafolder is the path to a folder on your local machine, where PySPI should save all the downloaded data needed for the analysis. The data that will be saved into this folder are the SPI data files as well as one housekeeping data file of SPI and one housekeeping data file of INTEGRAL per analyzed GRB. In total this adds up to roughly 30-70 MB per analyzed GRB. 
It is not recommended to use the same path for both environment variables.

You should also add these two line to your bashrc (or similar) file to automatically set these variables in every new terminal.

Now we are ready to go.

## Run Unit Test Locally

PySPI includes unit test to check that non of its functionality break in new versions. These run automatically for every push on GitHub via GitHub Actions. But you can also run the tests locally. To run the test you need to install pytest and pytest-cov.
```bash
pip install pytest pytest-cov
```
After this run
```bash
pytest -v
```
in the top level directory.

<!-- #endregion -->
