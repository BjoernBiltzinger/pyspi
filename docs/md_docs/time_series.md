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

# Light Curves

Setup to make the output clean for the docs:
```python
%%capture
from threeML import silence_logs
import warnings
warnings.filterwarnings("ignore")
silence_logs()
```

Gamma-Ray Bursts are transient sources with a typical duration between milliseconds and a few tens of seconds. Therefore they are nicely visible in light curves. In the following we will see how we can get the light curve of a real GRB as seen by an INTEGRAL/SPI detector.

First we have to define the rough time of the GRB.
```python
from astropy.time import Time
grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
```

Next we need to define the bounds of the energy bins we want to use.

```python
import numpy as np
ebounds = np.geomspace(20,8000,100)
```

Now we can construct the time series.

```python
from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
tsb = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{det}", 
                                        det, 
                                        ebounds, 
                                        grbtime, 
                                        sgl_type="both",
                                        )
```

We can now plot the light curves for visualization, in which we can clearly see a transient source in this case.

```python
tsb.view_lightcurve(-50,250)
```
