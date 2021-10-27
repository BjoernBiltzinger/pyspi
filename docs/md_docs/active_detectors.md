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

# Active Detectors

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

During the life of INTEGRAL/SPI several detectors stopped working correctly and were therefore disabled. In our analysis we need to take this into account, to not include a detector with 0 counts all the time and because the response for the surrounding detectors change when a detector is deactivated. 

With PySPI you can calculate for a given time, which detectors are active and which response version is valid at that time.

```python
time_string = "051212 205010" #"YYMMDD HHMMSS"; astropy time object also possible
```

To get the active single detectors for this time use:

```python
from pyspi.utils.livedets import get_live_dets
ld = get_live_dets(time_string, event_types="single")
print(f"Active detectors: {ld}")
```

It is also possible to plot the same information visually. This shows the detector plane, and all the inactive detectors at the given time are colored red.
```python
from pyspi.io.plotting.spi_display import SPI
s = SPI(time=time_string)
fig = s.plot_spi_working_dets()
```

Also the response version at that time can be calculated.
```python
from pyspi.utils.function_utils import find_response_version
v = find_response_version(time_string)
print(f"Response version number: {v}")
```
