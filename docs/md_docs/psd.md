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

# Electronic Noise Region

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

Since many years it is known that there are spurious events in the SPI data around ~1.5 MeV. 
A paper from [Roques & Jourdain](https://arxiv.org/abs/1811.06391) gives an explanation for this problem. Luckily this problem exists only in the events that only triggered the analog front-end electronics (AFEE). The events that trigger in addition the pulse shape discrimination electronics (PSD) do not show this problem. According to [Roques & Jourdain](https://arxiv.org/abs/1811.06391), one should therefore use the PSD events whenever this is possible, which is for events between ~500 and 2500 keV (the precise boundaries were changed during the mission time a few times). In the following the events that trigger both the AFEE and PSD are called "PSD events" and the other normal "single events", even thought the PSD events are of course also single events.

To account for this problem in out analysis we can construct plugins for the "PSD events" and the "single events" and use only the events with the correct flags, when we construct the time series.

Let's check the difference between the PSD and the normal single events, to see the effect in real SPI data. 

First we define the time and the energy bins we want to use. Then we construct the time series for the three cases:
1. Only the events that trigger AFEE and not PSD
2. Only the events that trigger AFEE and PSD
3. All the single events

```python
from astropy.time import Time
import numpy as np
from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
ebounds = np.geomspace(20,8000,300)
det = 0

from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
tsb_sgl = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{det}", 
    det, 
    ebounds, 
    grbtime, 
    sgl_type="sgl",
    )
    
tsb_psd = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{det}", 
    det, 
    ebounds, 
    grbtime, 
    sgl_type="psd",
    )

tsb_both = TimeSeriesBuilderSPI.from_spi_grb(f"SPIDet{det}", 
    det, 
    ebounds, 
    grbtime, 
    sgl_type="both",
    )
```

We can check the light curves for all three cases.

```python
print("Only AFEE:")
fig = tsb_sgl.view_lightcurve(-100,300)
```
```python
print("AFFE and PSD trigger:")
fig = tsb_psd.view_lightcurve(-100,300)
```
```python
print("Both Combined:")
fig = tsb_both.view_lightcurve(-100,300)
```

We can see that the PSD event light curve has way less counts. This is due to the fact, that the PSD trigger only starts working at energies > 400 keV.

Next we can get the time integrated counts per energy channel.

```python
tstart = -500
tstop = 1000
counts_sgl = tsb_sgl.time_series.count_per_channel_over_interval(tstart, tstop)
counts_psd = tsb_psd.time_series.count_per_channel_over_interval(tstart, tstop)
counts_both = tsb_both.time_series.count_per_channel_over_interval(tstart, tstop)
```

We can now plot the counts as a function of the energy channel energies
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1,1)
ax.step(ebounds[1:], counts_sgl, label="Only AFEE")
ax.step(ebounds[1:], counts_psd, label="AFEE and PSD")
ax.step(ebounds[1:], counts_both, label="All")
ax.set_xlabel("Detected Energy [keV]")
ax.set_ylabel("Counts")
ax.set_xlim(20,3500)
ax.set_yscale("log")
ax.legend();
```

Several features are visible. 
1. A sharp cutoff for at small energies for the PSD events, which is due to the low energy threshold in the PSD electronics. 
2. For energies>~2700 keV the PSD events are again decreasing faster than the other events
3. The fraction of PSD events to all the single events between ~500 and ~2700 keV is very stable and can be explained by an additional dead time for the PSD electronics
4. In the non PSD events we see a peak at ~ 1600 keV that is not visible in the PSD events. This is the so called electronic noise, which consists of spurious events.

