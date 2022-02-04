# GHSL Land Cover Service
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

**GHS-LC-SERV** (**GHS**L **L**and **C**over **Serv**ice) is an end-to-end, fully automated Earth Observation processing and analysis pipeline for generating custom [land cover maps](https://land.copernicus.eu/global/products/lc) from [Sentinel-2](https://sentinel.esa.int/web/sentinel/missions/sentinel-2) data archives.

## Quick start

The classification can be done on S2 tiles (files zip or SAFE directories) or on S2 mosaics (GeoTIFF format).

```python
from pathlib import Path
from ghslc import ghslc

# Sentinel 2 file to classify
s2_file = Path('S2A_MSIL1C_20191210T101411_N0208_R022_T32TQM_20191210T104357.zip')

# Training configuration as yaml file
training_file = Path('training_CGLS.yml')

# Target classes to extract from the classification
target_classes = [
    [80, 200],  # Permanent water bodies
    [111, 112, 113, 114, 115, 116, 121, 122, 123, 124, 125, 126],  # Forests
    40,  # Cultivated and managed vegetation/agriculture (cropland)
    50,  # Urban / built up
]

# Output folder
output = Path('/tmp')

results = ghslc.generate_classification_from_safe(
    filesafe=s2_file,
    workspace=output,
    training=training_file,
    classes=target_classes,
)
```


## Install
The easier way to install dependencies is using [conda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda env create -f environment.yml
```

Activate the conda environment:
```bash
conda activate ghslc
```

Install ghslc package:
```bash
pip install ghslc
```


## Test

Simply install [tox](https://tox.wiki/en/latest/#what-is-tox) with:
```bash
pip install tox
```

Then type:
```bash
tox
```

This will build, install and test the project in a dedicate environment with all needed dependencies.

The tests could take around 40 minutes to complete for each python version.

# Licensing
This project is licensed under the [GPLv3](http://www.gnu.org/licenses/gpl-3.0.html) License.

Copyright (c) 2021, [European Commission](https://ec.europa.eu/), Joint Research Centre. All rights reserved.
