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
There is currently no PyPI version of this software package, so you have to install it from Github. To to this run
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
wget https://grb.mpe.mpg.de/pyspi_datafolder
```
Now unzip the folder and move it to this location in the package: pyspi/pyspi/data (overwrite the data folder if there is already one)

You can check if the installation worked by running the crab_fit.ipynb notebook in the example folder.
