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
PySPI can be installed via pip.
```bash
pip install py-spi
```

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
Now you can install PySpi with
```bash
python setup.py install
```

There are a few large data files for the background model and the response that are not included in the Github repository. To get these data files run
```bash
cd pyspi
wget https://grb.mpe.mpg.de/pyspi_datafolder && unzip -o data.zip
```

Now you can delete the downloaded zip folder
```bash
rm -f data.zip
```

## Additional Data Files

There are a few large data files for the background model and the response that are not included in the Github repository. To get these data files run and specify the path where this data folder should be stored on your local machine.
```bash
wget https://grb.mpe.mpg.de/pyspi_datafolder && unzip data.zip && mv data /path/to/internal/data && rm -f data.zip
```

## Environment Variables

Next you have to set two environment variable. One to define the path to the folder of the external data like the different SPI data files that will be downloaded by PySPI and one to define the path to the internal data folder we downloaded earlier.
```bash
export PYSPI=/path/to/external/datafolder
export PYSPI_PACKAGE_DATA=/path/to/internal/data
```

You should add these two line to your bashrc file to automatically set this variable in every new terminal.

Now we are ready to go.

<!-- #endregion -->
