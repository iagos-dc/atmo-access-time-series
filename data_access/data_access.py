"""
This module provides a unified API for access to metadata and datasets from ACTRIS, IAGOS and ICOS RI's
and is an abstract layer over a lower-level data access API implemented in the package atmoaccess_data_access:
https://github.com/iagos-dc/atmo-access-data-access
"""
import pathlib
import pandas as pd
import functools

from atmoaccess_data_access import query_iagos, query_icos, query_actris

import config
import log
from . import helper

from log import logger


_QUERY_MODULE_BY_RI = {
    'icos': query_icos,
    'actris': query_actris,
    'iagos': query_iagos,
}

CACHE_DIR = pathlib.Path(config.APP_CACHE_DIR)
CACHE_DIR.mkdir(exist_ok=True)

# open an ACTRIS-specific cache; allows for an efficient ACTRIS metadata retrieval
query_actris._open_cache(CACHE_DIR / 'actris-cache.tmp')
query_icos._open_cache(CACHE_DIR / 'icos-cache.pkl')

_RIS = ('actris', 'iagos', 'icos')
_GET_DATASETS_BY_RI = dict()


# mapping from ECV names to short variable names (used for time-line graphs)
# must be updated on adding new RI's!
VARIABLES_MAPPING = {
    'Aerosol Optical Properties': 'AOP',
    'Aerosol Chemical Properties': 'ACP',
    'Aerosol Physical Properties': 'APP',
    'Pressure (surface)': 'AP',
    'Surface Wind Speed and direction': 'WSD',
    'Wind speed and direction (upper-air)': 'WSDu',
    'Temperature (near surface)': 'AT',
    'Temperature (upper-air)': 'ATu',
    'Water Vapour (surface)': 'RH',
    'Water Vapour (upper-air)': 'RHu',
    'Carbon Dioxide': 'CO2',
    'Carbon Monoxide': 'CO',
    'Methane': 'CH4',
    'Nitrous Oxide': 'N2O',
    # 'Carbon Dioxide, Methane and other Greenhouse gases': 'GG',
    'NO2': 'NO2',
    'Ozone': 'O3',
    'Cloud Properties': 'ClP',
}

var_codes_by_ECV = pd.Series(VARIABLES_MAPPING, name='code')
ECV_by_var_codes = pd.Series({v: k for k, v in VARIABLES_MAPPING.items()}, name='ECV')
_image_of_ecv2code = lambda ecvs: sorted(helper.image_of_dict(ecvs, VARIABLES_MAPPING))


@functools.lru_cache()
def _get_stations(ris):
    """
    Provide pandas DataFrame with
    :param ris: tuple of RIs; cannot be a list due to caching!
    :return: pandas.DataFrame
    """
    ris = [ri.lower() for ri in ris]
    stations_dfs = []
    for ri, ri_query_module in _QUERY_MODULE_BY_RI.items():
        if ri not in ris:
            continue
        cache_path = CACHE_DIR / f'stations_{ri}.json'
        try:
            try:
                stations_df = pd.read_json(cache_path, orient='table')
            except FileNotFoundError:
                stations = ri_query_module.get_list_platforms()
                if ri == 'iagos':
                    # add IAGOS regions
                    regions = ri_query_module.get_list_regions()
                    for region in regions:
                        region['longitude'] = 0.5 * (region['longitude_min'] + region['longitude_max'])
                        region['latitude'] = 0.5 * (region['latitude_min'] + region['latitude_max'])
                        region['is_region'] = True
                        region['URI'] = region['url']
                        del region['url']
                    stations.extend(regions)

                stations_df = pd.DataFrame.from_dict(stations)
                stations_df.to_json(cache_path, orient='table', indent=2)

            stations_df['RI'] = ri.upper()
            stations_dfs.append(stations_df)
        except Exception as e:
            logger().exception(f'getting {ri.upper()} stations failed', exc_info=e)

    all_stations_df = pd.concat(stations_dfs, ignore_index=True)
    all_stations_df['short_name_RI'] = all_stations_df['short_name'] + ' (' + all_stations_df['RI'] + ')'
    all_stations_df['idx'] = all_stations_df.index

    if 'is_region' not in all_stations_df:
        all_stations_df['is_region'] = False
    else:
        all_stations_df['is_region'] = all_stations_df['is_region'].fillna(False)

    return all_stations_df


def get_stations():
    """
    For each ACTRIS, IAGOS and ICOS station (for the moment it is ICOS only).
    :return: pandas Dataframe with stations data; it has the following columns:
    'URI', 'short_name', 'long_name', 'latitude', 'longitude', 'altitude', 'RI', 'short_name_RI', 'idx',
    'longitude_min', 'longitude_max', 'latitude_min', 'latitude_max', 'is_region'
    A sample record is:
        'URI': 'http://meta.icos-cp.eu/resources/stations/AS_BIR',
        'short_name': 'BIR',
        'long_name': 'Birkenes',
        'latitude': 58.3886,
        'longitude': 8.2519,
        'altitude': 219.0,
        'RI': 'ICOS',
        'short_name_RI': 'BIR (ICOS)',
        'idx': 2,
        'longitude_min': NaN,
        'longitude_max': NaN,
        'latitude_min': NaN,
        'latitude_max': NaN,
        'is_region': False,
    """
    return _get_stations(_RIS)


#@log_exectime
#@log_profiler_info()
def get_datasets(variables=None, station_codes=None, ris=None):
    """
    Provide metadata of datasets selected according to the provided criteria.
    :param variables: list of str or None; list of Essential Climate Variables (use the keys of the dictionary VARIABLES_MAPPING)
    :param station_code: list or None
    :param ris: list or None; must correspond to station_codes
    :return: pandas.DataFrame with columns: 'title', 'url', 'ecv_variables', 'platform_id', 'RI', 'selector', 'var_codes',
     'ecv_variables_filtered', 'var_codes_filtered',
     'time_period_start', 'time_period_end', 'platform_id_RI';
    e.g. for the call get_datasets(['Pressure (surface)', 'Temperature (near surface)'] one gets a dataframe with an example row like:
         'title': 'ICOS_ATC_L2_L2-2021.1_GAT_2.5_CTS_MTO.zip',
         'url': [{'url': 'https://meta.icos-cp.eu/objects/0HxLXMXolAVqfcuqpysYz8jK', 'type': 'landing_page'}],
         'ecv_variables': ['Pressure (surface)', 'Surface Wind Speed and direction', 'Temperature (near surface)', 'Water Vapour (surface)'],
         'platform_id': 'GAT',
         'RI': 'ICOS',
         'selector': NaN,
         'var_codes': ['AP', 'AT', 'RH', 'WSD'],
         'ecv_variables_filtered': ['Pressure (surface)', 'Temperature (near surface)'],
         'var_codes_filtered': 'AP, AT',
         'time_period_start': Timestamp('2016-05-10 00:00:00+0000', tz='UTC'),
         'time_period_end': Timestamp('2021-01-31 23:00:00+0000', tz='UTC'),
         'platform_id_RI': 'GAT (ICOS)'
    """
    if variables is None:
        # take all variables
        variables = list(VARIABLES_MAPPING)
    else:
        variables = list(variables)

    if station_codes is None or ris is None:
        # take all stations
        stations_df = get_stations()
        station_codes = stations_df['short_name'].to_list()
        ris = stations_df['RI'].to_list()

    assert len(station_codes) == len(ris), (f'station_codes and ris must have the same length; '
                                            f'got {station_codes=} of {len(station_codes)=} and {ris=} of {len(ris)=}')

    datasets_dfs = []
    for ri, get_ri_datasets in _GET_DATASETS_BY_RI.items():
        RI = ri.upper()
        try:
            station_codes_for_ri = [code for code, _ri in zip(station_codes, ris) if _ri.upper() == RI]
            df = get_ri_datasets(variables, station_codes_for_ri)
        except Exception as e:
            logger().exception(f'getting datasets for {RI} failed', exc_info=e)
            continue
        if df is not None:
            df['RI'] = RI
            datasets_dfs.append(df)

    if not datasets_dfs:
        return None
    datasets_df = pd.concat(datasets_dfs, ignore_index=True)

    datasets_df = filter_datasets_on_vars(datasets_df, variables)

    datasets_df['var_codes'] = datasets_df['ecv_variables'].map(_image_of_ecv2code)
    datasets_df['time_period_start'] = datasets_df['time_period'].map(lambda x: x[0])
    datasets_df['time_period_end'] = datasets_df['time_period'].map(lambda x: x[1])
    datasets_df['platform_id_RI'] = datasets_df['platform_id'] + ' (' + datasets_df['RI'] + ')'

    return datasets_df.drop(columns=['time_period']).rename(columns={'urls': 'url'})


@log.log_args
def _get_actris_datasets(variables, station_codes):
    datasets = query_actris.query_datasets(station_codes, variables_list=variables)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)
    return datasets_df


def _get_icos_datasets(variables, station_codes):
    _datasets = query_icos.query_datasets(station_codes, variables_list=variables)
    if not _datasets:
        return None
    datasets = []
    for dataset in _datasets:
        if len(dataset['urls']) > 0:
            datasets.append(dataset)
    if not datasets:
        return None

    datasets_df = pd.DataFrame.from_dict(datasets)

    # fix title for ICOS datasets: remove time span, which is after last comma
    datasets_df['title'] = datasets_df['title'].map(lambda s: ','.join(s.split(',')[:-1]))

    return datasets_df


def _get_iagos_datasets(variables, station_codes):
    _datasets = query_iagos.query_datasets(station_codes, variables_list=variables)
    datasets = []
    for dataset in _datasets:
        if len(dataset['urls']) > 0:
            datasets.append(dataset)
    if not datasets:
        return None

    datasets_for_layers = []
    variables = set(variables)
    for ds in datasets:
        title = ds['title']
        urls = ds['urls']
        time_period = ds['time_period']
        platform_id = ds['platform_id']
        ecv_for_all_layers = ds['ecv_variables']
        ecv_by_layer = ds.get('ecv_variables_by_layer', {})
        for layer in ds['layers']:
            ecv = ecv_by_layer.get(layer, ecv_for_all_layers)
            if variables.isdisjoint(ecv):
                continue
            ds_for_layer = {
                'title': f'{title} in {layer}',
                'urls': urls,
                'ecv_variables': ecv,
                'time_period': time_period,
                'platform_id': platform_id,
                'selector': f'layer={layer}'
            }
            datasets_for_layers.append(ds_for_layer)

    if not datasets_for_layers:
        return None

    return pd.DataFrame.from_dict(datasets_for_layers)


def filter_datasets_on_vars(datasets_df, ecvs):
    """
    Filter datasets which have at least one variables in common with ecvs and adds two new columns to datasets_df:
     'ecv_variables_filtered' and 'var_codes_filtered'
    :param datasets_df: pandas.DataFrame with datasets metadata (in the format returned by get_datasets function)
    :param ecvs: list of str; ECV names to filter on
    :return: pandas.DataFrame
    """
    ecvs = set(ecvs)
    datasets_df['ecv_variables_filtered'] = [
        sorted(ecvs.intersection(dataset_ecvs)) for dataset_ecvs in datasets_df['ecv_variables']
    ]
    var_codes_filtered = datasets_df['ecv_variables_filtered'].map(_image_of_ecv2code)
    datasets_df['var_codes_filtered'] = [', '.join(codes) for codes in var_codes_filtered]
    mask = datasets_df['ecv_variables_filtered'].map(len) > 0
    return datasets_df[mask]


def _infer_ts_frequency(da):
    dt = da.dropna('time', how='all')['time'].diff('time').to_series().reset_index(drop=True)
    if len(dt) > 0:
        freq = dt.value_counts().index[0] #.sort_index().index[0]
    else:
        freq = pd.Timedelta(0)
    return freq


def read_dataset(ri, url, ds_metadata, selector=None):
    ri = ri.lower()
    ds = _QUERY_MODULE_BY_RI[ri].read_dataset(url, variables_list=ds_metadata['ecv_variables_filtered'])
    #elif ri == 'iagos':
    if ds is not None and selector is not None:
        dim, coord = selector.split('=')
        coord = coord.split(':')
        if len(coord) == 1:
            coord, = coord
        elif len(coord) == 2 or len(coord) == 3:
            coord = slice(*coord)
        else:
            raise ValueError(f'bad selector={selector}')
        # if ds[dim].dtype is not object/str, need to convert coord to the dtype
        ds = ds.sel({dim: coord})

    if ds is not None:
        # infer time-series frequency
        for v, da in ds.items():
            freq = _infer_ts_frequency(da)
            if pd.Timedelta(27, 'D') <= freq <= pd.Timedelta(32, 'D'):
                freq = '1M'
            elif pd.Timedelta(1, 'D') - pd.Timedelta(60, 's') <= freq <= pd.Timedelta(1, 'D') + pd.Timedelta(60, 's'):
                freq = '1D'
            elif freq % pd.Timedelta(1, 'H') <= pd.Timedelta(5, 's') or pd.Timedelta(1, 'H') - freq % pd.Timedelta(1, 'H') <= pd.Timedelta(5, 's'):
                hours = (freq + pd.Timedelta(5, 's')) // pd.Timedelta(1, 'H')
                freq = f'{hours}H'
            else:
                freq = f'{int(freq.total_seconds())}s'
            da.attrs['_aats_freq'] = freq

    return ds


_GET_DATASETS_BY_RI.update(zip(_RIS, (_get_actris_datasets, _get_iagos_datasets, _get_icos_datasets)))
