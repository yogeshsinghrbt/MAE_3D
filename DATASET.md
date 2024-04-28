# Data Preparation
## The pre-processing can be summarized into following steps:

## 1. Rescaling
```python
import numpy as np

def rescale_volume(volume):
    minval = np.percentile(volume, 2)
    maxval = np.percentile(volume, 98)
 
    volume = np.clip(volume, minval, maxval)
    volume = ((volume - minval) / (maxval - minval))

    return volume

```

## 2. Resizing
```python
import skimage.transform as st

volume = st.resize(volume, [224, 224, 224], order=3, anti_aliasing=True, preserve_range=True)
```

### Note : we extract frames from the volume on the fly during training
