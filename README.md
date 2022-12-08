# atmo-access-time-series

<img align="right" width="200" src="https://www7.obs-mip.fr/wp-content-aeris/uploads/sites/82/2021/03/ATMO-ACCESS-Logo-final_horizontal-payoff-grey-blue.png">

Time series analysis service for ATMO-ACCESS project


## Installation

### Clone the git repository

```sh
git clone https://github.com/pawel-wolff/atmo-access-time-series
cd atmo-access-time-series
```

### Install python environment

- Using conda:

```sh
conda env create -f environment.yml
conda activate aats
```

- Using pip:

```sh
pip install -r requirements.txt
```


## Deployment at localhost

```sh
python app.py
```

Open a web browser and put `http://0.0.0.0:8050/` in the address bar.