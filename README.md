# atmo-access-time-series
Time series analysis service for ATMO-ACCESS project

[![Binder](https://notebooks.gesis.org/binder/badge_logo.svg)](https://mybinder.org/v2/gh/pawel-wolff/atmo-access-time-series/HEAD?urlpath=/tree/app.ipynb)

## Installation
```sh
git clone https://github.com/pawel-wolff/atmo-access-time-series
cd atmo-access-time-series
conda env create -f environment.yml
conda activate aa-time-series-env
```

## Deployment
Deployment in the stand-alone mode:
```sh
python app.py
```

Deployment in a Jupyter Notebook:
```sh
jupyter notebook
```
then open `app.ipynb` in the notebook and run all cells (can use >> button to that end).

If you need to change the application configuration, modify this part of the code (somewhere at the beginning of the script):
```python
RUNNING_IN_BINDER = False   # for running in Binder change it to True
app_conf = {'mode': 'external', 'debug': True}  # for running inside a Jupyter notebook change 'mode' to 'inline'
if RUNNING_IN_BINDER:
    JupyterDash.infer_jupyter_proxy_config()
else:
    app_conf.update({'host': 'localhost', 'port': 9235})
```
