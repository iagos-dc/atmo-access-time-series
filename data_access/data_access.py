"""
This module provides a unified API for access to metadata and datasets from ACTRIS, IAGOS and ICOS RI's.
At the moment, the access to ICOS RI is implemented here.
"""

import pkg_resources
import numpy as np
import pandas as pd
import xarray as xr
import pathlib
import json
import itertools

from .common import CACHE_DIR, DATA_DIR
from . import helper

from log import logger, log_exectime

from . import query_actris
from . import query_iagos
from . import query_icos


LON_LAT_BBOX_EPS = 0.05  # epsilon of margin for selecting stations

# for caching purposes
# TODO: do it properly
_stations = None
_variables = None


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


def _get_iagos_regions():
    regions = {
        'WNAm': ('western North America', [-125, -105], [40, 60]),
        'EUS': ('the eastern United States', [-90, -60], [35, 50]),
        'NAt': ('the North Atlantic', [-50, -20], [50, 60]),
        'Eur': ('Europe', [-15, 15], [45, 55]),
        'WMed': ('the western Mediterranean basin', [-5, 15], [35, 45]),
        'MidE': ('the Middle East', [25, 55], [35, 45]),
        'Sib': ('Siberia', [40, 120], [50, 65]),
        'NEAs': ('the northeastern Asia', [105, 145], [30, 50]),
    }
    short_name = list(regions)
    long_name = list(v[0] for v in regions.values())
    longitude_min, longitude_max = zip(*(v[1] for v in regions.values()))
    latitude_min, latitude_max = zip(*(v[2] for v in regions.values()))
    df = pd.DataFrame.from_dict({
        'short_name': short_name,
        'long_name': long_name,
        'longitude_min': longitude_min,
        'longitude_max': longitude_max,
        'latitude_min': latitude_min,
        'latitude_max': latitude_max,
    })
    df['longitude'] = 0.5 * (df['longitude_min'] + df['longitude_max'])
    df['latitude'] = 0.5 * (df['latitude_min'] + df['latitude_max'])
    df['is_region'] = True
    return df


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

                # add IAGOS regions
                stations_df = pd.concat([stations_df, _get_iagos_regions()], ignore_index=True)

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
    all_stations_df['marker_size'] = 7

    all_stations_df['is_region'] = all_stations_df['is_region'].fillna(False)

    return all_stations_df


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
        'is_region': False
    """
    global _stations
    if _stations is None:
        _stations = _get_stations()
    return _stations


def get_vars_long():
    """
    Provide a listing of RI's variables. For the same variable code there might be many records with different ECV names
    :return: pandas.DataFrame with columns: 'variable_name', 'ECV_name', 'std_ECV_name', 'code'; sample records are:
        'variable_name': 'co', 'ECV_name': 'Carbon Monoxide', 'std_ECV_name': 'Carbon Monoxide', 'code': 'CO';
        'variable_name': 'co', 'ECV_name': 'Carbon Dioxide, Methane and other Greenhouse gases', 'std_ECV_name': 'Carbon Monoxide', 'code': 'CO';
        'variable_name': 'co', 'ECV_name': 'co', 'std_ECV_name': 'Carbon Monoxide', 'code': 'CO';
    """
    global _variables
    if _variables is None:
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
    return _variables


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


#@log_exectime
#@log_profiler_info()
def get_datasets(variables, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Provide metadata of datasets selected according to the provided criteria.
    :param variables: list of str or None; list of variable standard ECV names (as in the column 'std_ECV_name' of the dataframe returned by get_vars function)
    :param lon_min: float or None
    :param lon_max: float or None
    :param lat_min: float or None
    :param lat_max: float or None
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
            df = get_ri_datasets(variables, bbox)
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
            v for v in vs if std_ECV_names_by_name[v].intersection(variables)
        )
        for vs in datasets_df['ecv_variables'].to_list()
    ]
    datasets_df['std_ecv_variables_filtered'] = [
        sorted(
            helper.image_of_dictOfLists(vs, std_ECV_names_by_name).intersection(variables)
        )
        for vs in datasets_df['ecv_variables'].to_list()
    ]
    req_var_codes = helper.image_of_dict(variables, VARIABLES_MAPPING)
    datasets_df['var_codes_filtered'] = [
        ', '.join(sorted(vc for vc in var_codes if vc in req_var_codes))
        for var_codes in datasets_df['var_codes'].to_list()
    ]
    # datasets_df['url'] = datasets_df['urls'].apply(lambda x: x[-1]['url'])  # now we take the last proposed url; TODO: see what should be a proper rule (first non-empty url?)
    datasets_df['time_period_start'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[0]))
    datasets_df['time_period_end'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[1]))
    datasets_df['platform_id_RI'] = datasets_df['platform_id'] + ' (' + datasets_df['RI'] + ')'

    # return datasets_df.drop(columns=['urls', 'time_period'])
    return datasets_df.drop(columns=['time_period']).rename(columns={'urls': 'url'})


#@log_exectime
#@log_profiler_info()
def get_datasets_old(variables, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Provide metadata of datasets selected according to the provided criteria.
    :param variables: list of str or None; list of variable standard ECV names (as in the column 'std_ECV_name' of the dataframe returned by get_vars function)
    :param lon_min: float or None
    :param lon_max: float or None
    :param lat_min: float or None
    :param lat_max: float or None
    :return: pandas.DataFrame with columns: 'title', 'url', 'ecv_variables', 'platform_id', 'RI', 'var_codes',
     'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes_filtered',
     'time_period_start', 'time_period_end', 'platform_id_RI';
    e.g. for the call get_datasets(['Pressure (surface)', 'Temperature (near surface)'] one gets a dataframe with an example row like:
         'title': 'ICOS_ATC_L2_L2-2021.1_GAT_2.5_CTS_MTO.zip',
         'url': [{'url': 'https://meta.icos-cp.eu/objects/0HxLXMXolAVqfcuqpysYz8jK', 'type': 'landing_page'}],
         'ecv_variables': ['Pressure (surface)', 'Surface Wind Speed and direction', 'Temperature (near surface)', 'Water Vapour (surface)'],
         'platform_id': 'GAT',
         'RI': 'ICOS',
         'var_codes': ['AP', 'AT', 'RH', 'WSD'],
         'ecv_variables_filtered': ['Pressure (surface)', 'Temperature (near surface)'],
         'std_ecv_variables_filtered': ['Pressure (surface)', 'Temperature (near surface)'],
         'var_codes_filtered': 'AP, AT',
         'time_period_start': Timestamp('2016-05-10 00:00:00+0000', tz='UTC'),
         'time_period_end': Timestamp('2021-01-31 23:00:00+0000', tz='UTC'),
         'platform_id_RI': 'GAT (ICOS)'
    """
    if variables is None:
        variables = []
    if None in [lon_min, lon_max, lat_min, lat_max]:
        bbox = []
    else:
        bbox = [lon_min, lat_min, lon_max, lat_max]

    datasets_dfs = []
    for ri, get_ri_datasets in _GET_DATASETS_BY_RI.items():
        # get_ri_datasets = log_exectime(get_ri_datasets)
        try:
            df = get_ri_datasets(variables, bbox)
        except Exception as e:
            logger().exception(f'getting datasets for {ri.upper()} failed', exc_info=e)
            continue
        if df is not None:
            datasets_dfs.append(df)

    if not datasets_dfs:
        return None
    datasets_df = pd.concat(datasets_dfs, ignore_index=True)

    vars_long = get_vars_long()
    def var_names_to_var_codes(var_names):
        # TODO: performance issues
        var_names = np.unique(var_names)
        codes1 = vars_long[['ECV_name', 'code']].join(pd.DataFrame(index=var_names), on='ECV_name', how='inner')
        codes1 = codes1['code'].unique()
        codes2 = vars_long[['variable_name', 'code']].join(pd.DataFrame(index=var_names), on='variable_name', how='inner')
        codes2 = codes2['code'].unique()
        codes = np.concatenate([codes1, codes2])
        return np.sort(np.unique(codes)).tolist()
    def var_names_to_std_ecv_by_var_name(var_names):
        # TODO: performance issues
        var_names = np.unique(var_names)
        std_ECV_names1 = vars_long[['ECV_name', 'std_ECV_name']].join(pd.DataFrame(index=var_names), on='ECV_name', how='inner')
        std_ECV_names1 = std_ECV_names1.rename(columns={'ECV_name': 'name'}).drop_duplicates(ignore_index=True)
        std_ECV_names2 = vars_long[['variable_name', 'std_ECV_name']].join(pd.DataFrame(index=var_names), on='variable_name', how='inner')
        std_ECV_names2 = std_ECV_names2.rename(columns={'variable_name': 'name'}).drop_duplicates(ignore_index=True)
        std_ECV_names = pd.concat([std_ECV_names1, std_ECV_names2], ignore_index=True).drop_duplicates(ignore_index=True)
        return std_ECV_names
    datasets_df['var_codes'] = datasets_df['ecv_variables'].apply(lambda var_names: var_names_to_var_codes(var_names))
    datasets_df['ecv_variables_filtered'] = datasets_df['ecv_variables'].apply(lambda var_names:
                                                                               var_names_to_std_ecv_by_var_name(var_names)\
                                                                               .join(pd.DataFrame(index=variables), on='std_ECV_name', how='inner')['name']\
                                                                               .sort_values()\
                                                                               .tolist())
    datasets_df['std_ecv_variables_filtered'] = datasets_df['ecv_variables'].apply(lambda var_names:
                                                                                   [v for v in var_names_to_std_ecv_by_var_name(var_names)['std_ECV_name'].sort_values().tolist() if v in variables])
    req_var_codes = set(VARIABLES_MAPPING[v] for v in variables if v in VARIABLES_MAPPING)
    datasets_df['var_codes_filtered'] = datasets_df['var_codes']\
        .apply(lambda var_codes: ', '.join(sorted(vc for vc in var_codes if vc in req_var_codes)))

    # datasets_df['url'] = datasets_df['urls'].apply(lambda x: x[-1]['url'])  # now we take the last proposed url; TODO: see what should be a proper rule (first non-empty url?)
    datasets_df['time_period_start'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[0]))
    datasets_df['time_period_end'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[1]))
    datasets_df['platform_id_RI'] = datasets_df['platform_id'] + ' (' + datasets_df['RI'] + ')'

    # return datasets_df.drop(columns=['urls', 'time_period'])
    return datasets_df.drop(columns=['time_period']).rename(columns={'urls': 'url'})


def _get_actris_datasets(variables, bbox):
    datasets = query_actris.query_datasets(variables=variables, temporal_extent=[], spatial_extent=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)

    # fix title for ACTRIS datasets: remove time span
    datasets_df['title'] = datasets_df['title'].str.slice(stop=-62)

    datasets_df['RI'] = 'ACTRIS'
    return datasets_df


def _get_icos_datasets(variables, bbox):
    datasets = query_icos.query_datasets(variables=variables, temporal=[], spatial=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)

    # fix title for ICOS datasets: remove time span, which is after last comma
    datasets_df['title'] = datasets_df['title'].map(lambda s: ','.join(s.split(',')[:-1]))

    datasets_df['RI'] = 'ICOS'
    return datasets_df


_iagos_catalogue_df = None

def _get_iagos_datasets_catalogue():
    global _iagos_catalogue_df
    if _iagos_catalogue_df is None:
        url = DATA_DIR / 'catalogue.json'
        with open(url, 'r') as f:
            md = json.load(f)
        _iagos_catalogue_df = pd.DataFrame.from_records(md)
    return _iagos_catalogue_df


def _get_iagos_datasets(variables, bbox):
    variables = set(variables)
    df = _get_iagos_datasets_catalogue()
    variables_filter = df['ecv_variables'].map(lambda vs: bool(variables.intersection(vs)))
    lon_min, lat_min, lon_max, lat_max = bbox
    bbox_filter = (df['longitude'] >= lon_min - LON_LAT_BBOX_EPS) & (df['longitude'] <= lon_max + LON_LAT_BBOX_EPS) & \
                  (df['latitude'] >= lat_min - LON_LAT_BBOX_EPS) & (df['latitude'] <= lat_max + LON_LAT_BBOX_EPS)
    df = df[variables_filter & bbox_filter].explode('layer', ignore_index=True)
    df['title'] = df['title'] + ' in ' + df['layer']
    df['selector'] = 'layer=' + df['layer']
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


def filter_datasets_on_vars(datasets_df, var_codes):
    """
    Filter datasets which have at least one variables in common with var_codes.
    :param datasets_df: pandas.DataFrame with datasets metadata (in the format returned by get_datasets function)
    :param var_codes: list of str; variables codes
    :return: pandas.DataFrame
    """
    var_codes = set(var_codes)
    mask = datasets_df['var_codes'].apply(lambda vc: not var_codes.isdisjoint(vc))
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
        iagos_data_path = DATA_DIR / 'iagos_L3_postprocessed'
        ds = xr.open_dataset(iagos_data_path / url)
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
        std_ecv_to_vcode = {
            'Carbon Monoxide': 'CO_mean',
            'Ozone': 'O3_mean',
        }
        vs = [std_ecv_to_vcode[v] for v in ds_metadata['std_ecv_variables_filtered']]
        ds = ds[vs].load()
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
