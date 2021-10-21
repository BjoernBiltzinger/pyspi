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
cd pyspi
wget https://grb.mpe.mpg.de/pyspi_datafolder && unzip -o data.zip
```

Next you have to set a environment variable to define the storage folder for the different data files that will be downloaded.
```bash
export PYSPI=/path/to/datafolder
```
You should add this line to your bashrc file to always automatically set this variable in a new terminal.

Now we are ready to go.

<!-- #endregion -->
