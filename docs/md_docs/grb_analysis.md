---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Analyse GRB data


Setup to make the output clean for the docs:
```python
%%capture
from threeML import silence_logs
import warnings
warnings.filterwarnings("ignore")
silence_logs()
import matplotlib.pyplot as plt
%matplotlib inline
from jupyterthemes import jtplot
jtplot.style(context="talk", fscale=1, ticks=True, grid=False)
```

The first thing we need to do, is to specify the time of the GRB. We do this by specifying a astropy time object or a string in the format YYMMDD HHMMSS.
```python
from astropy.time import Time
grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
#grbtime = "120711 024453" # works also
```

Next, we need to specify the output and input energy bins we want to use.
```python
import numpy as np
ein = np.geomspace(20,800,300)
ebounds = np.geomspace(20,400,30)
```

Due to detector failures there are several versions of the response for SPI. Therefore we have to find the version number for the time of the GRB and construct the base response object for this version.
```python
from pyspi.utils.function_utils import find_response_version
from pyspi.utils.response.spi_response_data import ResponseDataRMF
version = find_response_version(grbtime)
print(version)
rsp_base = ResponseDataRMF.from_version(version)
```

Now we can create the response object for detector 0 and set the position of the GRB, which we already know.
```python
from pyspi.utils.response.spi_response import ResponseRMFGenerator
from pyspi.utils.response.spi_drm import SPIDRM
det=0
ra = 94.6783
dec = -70.99905
drm_generator = ResponseRMFGenerator.from_time(grbtime, 
                                                det,
                                                ebounds, 
                                                ein,
                                                rsp_base)
sd = SPIDRM(drm_generator, ra, dec)
```

With this we can build a time series and we use all the single events in this case (PSD + non PSD; see section about electronic noise). To be able to convert the time series into 3ML plugins later, we need to assign them a response object.
```python
from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
tsb = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{det}", 
    det, 
    grbtime, 
    response=sd,
    sgl_type="both",
    )
```

Now we can have a look at the light curves from -50 to 150 seconds around the specified GRB time.
```python
fig = tsb.view_lightcurve(-50,150)
```

With this we can select the active time and some background time intervals.
```python
active_time = "65-75"
bkg_time1 = "-500--10"
bkg_time2 = "150-1000"
tsb.set_active_time_interval(active_time)
tsb.set_background_interval(bkg_time1, bkg_time2)
```
We can check if the selection and background fitting worked by looking again at the light curve
```python tags=["nbsphinx-thumbnail"]
fig = tsb.view_lightcurve(-50,150)
```
For the fit we of course want to use all the available detectors. So we first check which detectors were still working at that time.
```python
from pyspi.utils.livedets import get_live_dets
active_dets = get_live_dets(time=grbtime, event_types=["single"])
print(active_dets)
```

Now we loop over these detectors, build the times series, fit the background and construct the SPILikeGRB plugins which we can use in 3ML.
```python
from pyspi.SPILike import SPILikeGRB
from threeML import DataList
spilikes = []
for d in active_dets:
    drm_generator = ResponseRMFGenerator.from_time(grbtime, 
                                                    d,
                                                    ebounds, 
                                                    ein,
                                                    rsp_base)
    sd = SPIDRM(drm_generator, ra, dec)
    tsb = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{d}", 
                                                d,
                                                grbtime, 
                                                response=sd,
                                                sgl_type="both",
                                                )
    tsb.set_active_time_interval(active_time)
    tsb.set_background_interval(bkg_time1, bkg_time2)

    sl = tsb.to_spectrumlike()
    spilikes.append(SPILikeGRB.from_spectrumlike(sl,
                                                free_position=False))
datalist = DataList(*spilikes)
```

Now we have to specify a model for the fit. We use [astromodels](https://astromodels.readthedocs.io/en/latest/) for this.
```python
from astromodels import *
pl = Powerlaw()
pl.K.prior = Log_uniform_prior(lower_bound=1e-6, upper_bound=1e4)
pl.K.bounds = (1e-6, 1e4)
pl.index.set_uninformative_prior(Uniform_prior)
pl.piv.value = 200
ps = PointSource('GRB',ra=ra, dec=dec, spectral_shape=pl)

model = Model(ps)
```

Everything is ready to fit now! We make a Bayesian fit here with emcee
```python
from threeML import BayesianAnalysis
ba_spi = BayesianAnalysis(model, datalist)
ba_spi.set_sampler("emcee", share_spectrum=True)
ba_spi.sampler.setup(n_walkers=20, n_iterations=500)
ba_spi.sample()
```

We can inspect the fits with residual plots

```python
from threeML import display_spectrum_model_counts
fig = display_spectrum_model_counts(ba_spi, 
                                data_per_plot=5, 
                                source_only=True,
                                show_background=False,
                                model_cmap="viridis", 
                                data_cmap="viridis",
                                background_cmap="viridis")
```

and have a look at the spectrum

```python
from threeML import plot_spectra
fig = plot_spectra(ba_spi.results, flux_unit="keV/(s cm2)", ene_min=20, ene_max=600)
```
We can also get a summary of the fit and write the results to disk (see 3ML documentation).


It is also possible to localize GRBs with PySPI, to this we simply free the position of point source with:

```python
for s in spilikes:
    s.set_free_position(True)
datalist = DataList(*spilikes)
```
Initialize the Bayesian Analysis and start the sampling with MultiNest. To use MultiNest you need to install [pymultinest](https://github.com/JohannesBuchner/PyMultiNest) according to its [documentation](https://johannesbuchner.github.io/PyMultiNest/install.html). 
```python
import os
os.mkdir("./chains_grb_example")
ba_spi = BayesianAnalysis(model, datalist)
ba_spi.set_sampler("multinest")
ba_spi.sampler.setup(500, 
                    chain_name='./chains_grb_example/docsfit1_',
                    resume=False,
                    verbose=False)
ba_spi.sample()
```

We can use the 3ML features to create a corner plot for this fit:

```python
from threeML.config.config import threeML_config
threeML_config.bayesian.corner_style.show_titles = False
fig = ba_spi.results.corner_plot(components=["GRB.position.ra", "GRB.position.dec"])
```

When we compare the results for ra and dec, we can see that this matches with the position from [Swift-XRT for the same GRB (RA, Dec = 94.67830, -70.99905)](https://gcn.gsfc.nasa.gov/gcn/other/120711A.gcn3)
