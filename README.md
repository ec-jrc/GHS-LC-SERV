# GHSL Land Cover Service
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

**GHS-LC-SERV** (**GHS**L **L**and **C**over **Serv**ice) is an end-to-end processing and analysis pipeline aimed at the European and national stakeholders, providing them with a fully-automated tool for generating custom land cover maps from Sentinel-2 data archives. It consists of an Earth Observation processing system linked with a visualization tool allowing any user to generate, at his own premises and in an operational way, land cover maps tailored to his needs.

## Quick start


## Install
The easier way to install dependencies is using [conda](https://docs.conda.io/en/latest/miniconda.html):

```bash
conda create -f environment.yml
conda activate ghslc
```

Install ghslc package:
```bash
pip install -e .
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
