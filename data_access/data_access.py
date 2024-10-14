"""
This module provides a unified API for access to metadata and datasets from ACTRIS, IAGOS and ICOS RI's.
At the moment, the access to ICOS RI is implemented here.
"""

import numpy as np
import pandas as pd
import xarray as xr
import json
import itertools
import functools

from .common import CACHE_DIR, DATA_DIR
from . import helper

from log import logger

from . import query_actris
from . import query_iagos as query_iagos_old
from . import query_iagos2 as query_iagos
from . import query_icos


_RIS = ['actris', 'iagos', 'icos']
_GET_DATASETS_BY_RI = dict()


# mapping from standard ECV names to short variable names (used for time-line graphs)
# must be updated on adding new RI's!
VARIABLES_MAPPING = {
    'Aerosol Optical Properties': 'AOP',
    'Aerosol Chemical Properties': 'ACP',
    'Aerosol Physical Properties': 'APP',
    'Pressure (surface)': 'AP',
    'Surface Wind Speed and direction': 'WSD',
    'Temperature (near surface)': 'AT',
    'Water Vapour (surface)': 'RH',
    'Carbon Dioxide': 'CO2',
    'Carbon Monoxide': 'CO',
    'Methane': 'CH4',
    'Nitrous Oxide': 'N2O',
    'NO2': 'NO2',
    'Ozone': 'O3',
    'Cloud Properties': 'ClP',
}


_var_codes_by_ECV = pd.Series(VARIABLES_MAPPING, name='code')
_ECV_by_var_codes = pd.Series({v: k for k, v in VARIABLES_MAPPING.items()}, name='ECV')


def _get_ri_query_module_by_ri(ris=None):
    if ris is None:
        ris = _RIS
    else:
        ris = sorted(ri.lower() for ri in ris)
    ri_query_module_by_ri = {}
    for ri in ris:
        if ri == 'actris':
            ri_query_module_by_ri[ri] = query_actris
        elif ri == 'iagos':
            ri_query_module_by_ri[ri] = query_iagos
        elif ri == 'icos':
            ri_query_module_by_ri[ri] = query_icos
        else:
            raise ValueError(f'ri={ri}')
    return ri_query_module_by_ri


_ri_query_module_by_ri = _get_ri_query_module_by_ri()


def _get_stations(ris=None):
    ri_query_module_by_ri = _get_ri_query_module_by_ri(ris)
    stations_dfs = []
    for ri, ri_query_module in ri_query_module_by_ri.items():
        cache_path = CACHE_DIR / f'stations_{ri}.pkl'
        try:
            try:
                stations_df = pd.read_pickle(cache_path)
            except FileNotFoundError:
                stations = ri_query_module.get_list_platforms()
                if ri == 'iagos':
                    # add IAGOS regions
                    regions = ri_query_module.get_list_regions()
                    for region in regions:
                        region['longitude'] = 0.5 * (region['longitude_min'] + region['longitude_max'])
                        region['latitude'] = 0.5 * (region['latitude_min'] + region['latitude_max'])
                        region['is_region'] = True
                    stations.extend(regions)
                stations_df = pd.DataFrame.from_dict(stations)
                stations_df.to_pickle(cache_path)

            if ri == 'actris':
                stations_df = stations_df.rename(columns={'URI': 'uri', 'altitude': 'ground_elevation'})
                stations_df['RI'] = 'ACTRIS'
                stations_df['country'] = np.nan
                stations_df['theme'] = np.nan
                stations_dfs.append(stations_df)
            elif ri == 'iagos':
                stations_df = stations_df.rename(columns={'altitude': 'ground_elevation'})
                stations_df['RI'] = 'IAGOS'
                stations_df['uri'] = np.nan
                stations_df['country'] = np.nan
                stations_df['theme'] = np.nan
                stations_dfs.append(stations_df)
            elif ri == 'icos':
                for col in ['latitude', 'longitude', 'ground_elevation']:
                    stations_df[col] = pd.to_numeric(stations_df[col])
                stations_dfs.append(stations_df)
            else:
                raise ValueError(f'ri={ri}')
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


@functools.lru_cache()
def get_stations():
    """
    For each ACTRIS, IAGOS and ICOS station (for the moment it is ICOS only).
    :return: pandas Dataframe with stations data; it has the following columns:
    'uri', 'short_name', 'long_name', 'country', 'latitude', 'longitude', 'ground_elevation', 'RI', 'short_name_RI',
    'theme', 'idx',
    'longitude_min', 'longitude_max', 'latitude_min', 'latitude_max', 'is_region'
    A sample record is:
        'uri': 'http://meta.icos-cp.eu/resources/stations/AS_BIR',
        'short_name': 'BIR',
        'long_name': 'Birkenes',
        'country': 'NO',
        'latitude': 58.3886,
        'longitude': 8.2519,
        'ground_elevation': 219.0,
        'RI': 'ICOS',
        'short_name_RI': 'BIR (ICOS)',
        'theme': 'AS',
        'idx': 2,
        'longitude_min': NaN,
        'longitude_max': NaN,
        'latitude_min': NaN,
        'latitude_max': NaN,
        'is_region': False,
    """
    return _get_stations()


@functools.lru_cache()
def get_vars_long():
    """
    Provide a listing of RI's variables. For the same variable code there might be many records with different ECV names
    :return: pandas.DataFrame with columns: 'variable_name', 'ECV_name', 'std_ECV_name', 'code'; sample records are:
        'variable_name': 'co', 'ECV_name': 'Carbon Monoxide', 'std_ECV_name': 'Carbon Monoxide', 'code': 'CO';
        'variable_name': 'co', 'ECV_name': 'Carbon Dioxide, Methane and other Greenhouse gases', 'std_ECV_name': 'Carbon Monoxide', 'code': 'CO';
        'variable_name': 'co', 'ECV_name': 'co', 'std_ECV_name': 'Carbon Monoxide', 'code': 'CO';
    """
    # ri_query_module_by_ri = _get_ri_query_module_by_ri(['actris', 'icos'])
    variables_dfs = []
    for ri, ri_query_module in _ri_query_module_by_ri.items():
        cache_path = CACHE_DIR / f'variables_{ri}.pkl'
        try:
            try:
                variables_df = pd.read_pickle(cache_path)
            except FileNotFoundError:
                variables = ri_query_module.get_list_variables()
                variables_df = pd.DataFrame.from_dict(variables)
                variables_df.to_pickle(cache_path)
            variables_dfs.append(variables_df)
        except Exception as e:
            logger().exception(f'getting {ri.upper()} variables failed', exc_info=e)
    df = pd.concat(variables_dfs, ignore_index=True)
    df['std_ECV_name'] = df['ECV_name'].apply(lambda l: l[0])
    df = df.join(_var_codes_by_ECV, on='std_ECV_name')
    _variables = df.explode('ECV_name', ignore_index=True).drop_duplicates(keep='first', ignore_index=True)

    # sort _variables dataframe by 'code', 'std_ECV_name', 'ECV_name'
    # however, do not distinguish between lower and upper-case letters
    sortby_ori = ['code', 'std_ECV_name', 'ECV_name']
    sortby_upper = [f'_{col}_uppercase' for col in sortby_ori]
    for col_ori, col_upper in zip(sortby_ori, sortby_upper):
        _variables[col_upper] = _variables[col_ori].str.upper()
    _variables = _variables.sort_values(by=sortby_upper)
    _variables = _variables.drop(columns=sortby_upper)

    # _ = _variables.to_dict(orient='records')
    # _ = json.dumps(_, indent=2)
    # print(_)

    return _variables


@functools.lru_cache()
def get_vars():
    """
    Provide a listing of RI's variables.
    :return: pandas.DataFrame with columns: 'ECV_name', 'std_ECV_name', 'code'; a sample record is:
        'ECV_name': 'Carbon Dioxide, Methane and other Greenhouse gases',
        'std_ECV_name': 'Carbon Monoxide',
        'code': 'CO'
    """
    variables_df = get_vars_long().drop(columns=['variable_name'])
    return variables_df.drop_duplicates(subset=['std_ECV_name', 'ECV_name'], keep='first', ignore_index=True)


@functools.lru_cache()
def get_std_ECV_name_by_code():
    """
    Provides a dictionary variable code -> std ECV name
    :return: dict
    """
    std_ECV_name_by_code_series = get_vars()[['code', 'std_ECV_name']].drop_duplicates().set_index('code')['std_ECV_name']
    if not std_ECV_name_by_code_series.index.is_unique:
        logger().warning(f'std_ECV_name is not unique for a given variable code: {std_ECV_name_by_code_series}')
    return std_ECV_name_by_code_series.to_dict()


#@log_exectime
#@log_profiler_info()
def get_datasets(variables, station_codes=None, ris=None, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Provide metadata of datasets selected according to the provided criteria.
    :param variables: list of str or None; list of variable standard ECV names (as in the column 'std_ECV_name' of the dataframe returned by get_vars function)
    :param station_code: list or None
    :param ris: list or None; must correspond to station_codes
    :param lon_min: float or None (deprecated)
    :param lon_max: float or None (deprecated)
    :param lat_min: float or None (deprecated)
    :param lat_max: float or None (deprecated)
    :return: pandas.DataFrame with columns: 'title', 'url', 'ecv_variables', 'platform_id', 'RI', 'selector', 'var_codes',
     'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes_filtered',
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
         'std_ecv_variables_filtered': ['Pressure (surface)', 'Temperature (near surface)'],
         'var_codes_filtered': 'AP, AT',
         'time_period_start': Timestamp('2016-05-10 00:00:00+0000', tz='UTC'),
         'time_period_end': Timestamp('2021-01-31 23:00:00+0000', tz='UTC'),
         'platform_id_RI': 'GAT (ICOS)'
    """
    if variables is None:
        # take all variables
        variables = sorted(get_vars_long()['std_ECV_name'].unique())
    else:
        variables = list(variables)
    if None in [lon_min, lon_max, lat_min, lat_max]:
        bbox = []
    else:
        bbox = [lon_min, lat_min, lon_max, lat_max]

    datasets_dfs = []
    for ri, get_ri_datasets in _GET_DATASETS_BY_RI.items():
        # get_ri_datasets = log_exectime(get_ri_datasets)
        try:
            df = get_ri_datasets(variables, station_codes, ris, bbox)
        except Exception as e:
            logger().exception(f'getting datasets for {ri.upper()} failed', exc_info=e)
            continue
        if df is not None:
            datasets_dfs.append(df)

    if not datasets_dfs:
        return None
    datasets_df = pd.concat(datasets_dfs, ignore_index=True)#.reset_index()

    vars_long = get_vars_long()

    codes_by_ECV_name = helper.many2many_to_dictOfList(
        zip(vars_long['ECV_name'].to_list(), vars_long['code'].to_list())
    )
    codes_by_variable_name = helper.many2many_to_dictOfList(
        zip(vars_long['variable_name'].to_list(), vars_long['code'].to_list())
    )
    codes_by_name = helper.many2manyLists_to_dictOfList(
        itertools.chain(codes_by_ECV_name.items(), codes_by_variable_name.items())
    )
    datasets_df['var_codes'] = [
        sorted(helper.image_of_dictOfLists(vs, codes_by_name))
        for vs in datasets_df['ecv_variables'].to_list()
    ]

    datasets_df = filter_datasets_on_vars(datasets_df, variables)

    # datasets_df['url'] = datasets_df['urls'].apply(lambda x: x[-1]['url'])  # now we take the last proposed url; TODO: see what should be a proper rule (first non-empty url?)
    datasets_df['time_period_start'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[0], tz='UTC'))
    datasets_df['time_period_end'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[1], tz='UTC'))
    datasets_df['platform_id_RI'] = datasets_df['platform_id'] + ' (' + datasets_df['RI'] + ')'

    # return datasets_df.drop(columns=['urls', 'time_period'])
    return datasets_df.drop(columns=['time_period']).rename(columns={'urls': 'url'})


def _get_actris_datasets(variables, station_codes, ris, bbox):
    datasets = query_actris.query_datasets(variables=variables, temporal_extent=[], spatial_extent=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)

    # fix title for ACTRIS datasets: remove time span
    datasets_df['title'] = datasets_df['title'].str.slice(stop=-62)

    datasets_df['RI'] = 'ACTRIS'
    return datasets_df


def _get_icos_datasets(variables, station_codes, ris, bbox):
    datasets = query_icos.query_datasets(variables=variables, temporal=[], spatial=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)

    # fix title for ICOS datasets: remove time span, which is after last comma
    datasets_df['title'] = datasets_df['title'].map(lambda s: ','.join(s.split(',')[:-1]))

    datasets_df['RI'] = 'ICOS'
    return datasets_df


def _get_iagos_datasets(variables, station_codes, ris, bbox):
    variables = set(variables)
    if station_codes is None:
        _iagos_stations = _get_stations(['iagos'])
        station_codes = _iagos_stations['short_name']
        ris = _iagos_stations['RI']
    station_codes = list(station_codes)
    ris = list(ris)
    iagos_station_codes = [code for code, ri in zip(station_codes, ris) if ri == 'IAGOS']

    datasets = query_iagos.query_datasets_stations(iagos_station_codes, variables_list=variables)
    for dataset in datasets:
        dataset['urls'] = dataset['urls'][0]['url']
    df = pd.DataFrame.from_records(datasets)
    df['RI'] = 'IAGOS'
    variables_filter = df['ecv_variables'].map(lambda vs: bool(variables.intersection(vs)))
    df = df[variables_filter].explode('layers', ignore_index=True)
    df['title'] = df['title'] + ' in ' + df['layers']
    df['selector'] = 'layer=' + df['layers']
    df = df[['title', 'urls', 'ecv_variables', 'time_period', 'platform_id', 'RI', 'selector']]
    return df


def filter_datasets_on_stations(datasets_df, stations_short_name):
    """
    Filter datasets on stations by their short names.
    :param datasets_df: pandas.DataFrame with datasets metadata (in the format returned by get_datasets function)
    :param stations_short_name: list of str; short names of stations
    :return: pandas.DataFrame
    """
    return datasets_df[datasets_df['platform_id'].isin(stations_short_name)]


def filter_datasets_on_vars(datasets_df, std_ecv_names):
    """
    Filter datasets which have at least one variables in common with std_ecv_names.
    :param datasets_df: pandas.DataFrame with datasets metadata (in the format returned by get_datasets function)
    :param std_ecv_names: list of str; standard ECV variables names to filter on
    :return: pandas.DataFrame
    """
    vars_long = get_vars_long()

    std_ECV_names_by_ECV_name = helper.many2many_to_dictOfList(
        zip(vars_long['ECV_name'].to_list(), vars_long['std_ECV_name'].to_list()), keep_set=True
    )
    std_ECV_names_by_variable_name = helper.many2many_to_dictOfList(
        zip(vars_long['variable_name'].to_list(), vars_long['std_ECV_name'].to_list()), keep_set=True
    )
    std_ECV_names_by_name = helper.many2manyLists_to_dictOfList(
        itertools.chain(std_ECV_names_by_ECV_name.items(), std_ECV_names_by_variable_name.items()), keep_set=True
    )
    datasets_df['ecv_variables_filtered'] = [
        sorted(
            v for v in vs if std_ECV_names_by_name[v].intersection(std_ecv_names)
        )
        for vs in datasets_df['ecv_variables'].to_list()
    ]
    datasets_df['std_ecv_variables_filtered'] = [
        sorted(
            helper.image_of_dictOfLists(vs, std_ECV_names_by_name).intersection(std_ecv_names)
        )
        for vs in datasets_df['ecv_variables'].to_list()
    ]
    req_var_codes = helper.image_of_dict(std_ecv_names, VARIABLES_MAPPING)
    datasets_df['var_codes_filtered'] = [
        ', '.join(sorted(vc for vc in var_codes if vc in req_var_codes))
        for var_codes in datasets_df['var_codes'].to_list()
    ]

    mask = datasets_df['std_ecv_variables_filtered'].apply(len) > 0
    #print(list(datasets_df.platform_id))
    #print(datasets_df[mask & (datasets_df.platform_id == 'SZW')].iloc[0])
    return datasets_df[mask]


def _infer_ts_frequency(da):
    dt = da.dropna('time', how='all')['time'].diff('time').to_series().reset_index(drop=True)
    if len(dt) > 0:
        freq = dt.value_counts().sort_index().index[0]
    else:
        freq = pd.Timedelta(0)
    return freq


def read_dataset(ri, url, ds_metadata, selector=None):
    if isinstance(url, (list, tuple)):
        ds = None
        for single_url in url:
            ds = read_dataset(ri, single_url, ds_metadata)
            if ds is not None:
                break
        return ds

    if isinstance(url, dict):
        return read_dataset(ri, url['url'], ds_metadata)

    if not isinstance(url, str):
        raise ValueError(f'url must be str; got: {url} of type={type(url)}')

    # logger().info(f'ri={ri}, url={url}, ds_metadata=\n{ds_metadata}')

    ri = ri.lower()
    if ri == 'actris':
        ds = _ri_query_module_by_ri[ri].read_dataset(url, ds_metadata['ecv_variables_filtered'])
        # TODO: this is a temporary patch !!! to be done properly
        if len(ds.dims) > 1:
            non_time_dims = [d for d in ds.dims if d != 'time']
            non_time_dims_selector = {d: 0 for d in non_time_dims}
            ds = ds.isel(non_time_dims_selector, drop=True)
    elif ri == 'icos':
        ds = _ri_query_module_by_ri[ri].read_dataset(url)
        vars_long = get_vars_long()
        variables_names_filtered = list(vars_long.join(
            pd.DataFrame(index=ds_metadata['std_ecv_variables_filtered']),
            on='std_ECV_name',
            how='inner')['variable_name'].unique())
        variables_names_filtered = [v for v in ds if v in variables_names_filtered]
        ds_filtered = ds[['TIMESTAMP'] + variables_names_filtered].compute()
        ds = ds_filtered.assign_coords({'index': ds['TIMESTAMP']}).rename({'index': 'time'}).drop_vars('TIMESTAMP')
    elif ri == 'iagos':
        ds = _ri_query_module_by_ri[ri].read_dataset(url, variables_list=ds_metadata['std_ecv_variables_filtered'])
        if selector is not None:
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
    else:
        raise ValueError(f'unknown RI={ri}')

    res = {}
    if ds is not None:
        for v, da in ds.items():
            freq = _infer_ts_frequency(da)
            if pd.Timedelta(28, 'D') <= freq <= pd.Timedelta(31, 'D'):
                freq = '1M'
            else:
                freq = f'{int(freq.total_seconds())}s'
            da.attrs['_aats_freq'] = freq
            if freq != '0s':
                # da_resampled = da.resample({'time': freq}).asfreq()
                # da_resampled = da.resample({'time': freq}).interpolate()
                # da_resampled.attrs = dict(da.attrs)
                # res[v] = da_resampled
                res[v] = da
            else:
                res[v] = da

    #return res
    return ds


_GET_DATASETS_BY_RI.update(zip(_RIS, (_get_actris_datasets, _get_iagos_datasets, _get_icos_datasets)))
