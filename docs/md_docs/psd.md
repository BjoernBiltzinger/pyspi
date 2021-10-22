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
```

Since many years it is known that there are spurious events in the SPI data around ~1.5 MeV. 
A paper from [Roques & Jourdain](https://arxiv.org/abs/1811.06391) gives an explanation for this problem. Luckily this problem exists only in the events that only triggered the analog front-end electronics (AFEE). The events that trigger in addition the pulse shape discrimination electronics (PSD) do not show this problem. According to [Roques & Jourdain](https://arxiv.org/abs/1811.06391), one should therefore use the PSD events whenever this is possible, which is for events between ~500 and 2500 keV (the precise boundaries are were changed during the mission time a few times). In the following the events that trigger both the AFEE and PSD are called "PSD events" and the other normal "single events", even thought the PSD events are of course also single events.

To account for this problem in out analysis we can construct plugins for the "PSD events" and the "single events" and use only the events with the correct flags, when we construct the time series.

Let's check the difference between the PSD and the normal single events, to see the effect in real SPI data. 


```python
from astropy.time import Time
import numpy as np
from pyspi.utils.data_builder.time_series_builder import TimeSeriesBuilderSPI
grbtime = Time("2012-07-11T02:44:53", format='isot', scale='utc')
ebounds = np.geomspace(20,400,30)

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


