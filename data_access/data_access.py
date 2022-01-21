"""
This module provides a unified API for access to metadata and datasets from ACTRIS, IAGOS and ICOS RI's.
At the moment, the access to ICOS RI is implemented here.
"""

import numpy as np
import pandas as pd
import logging
import pathlib

from . import query_actris
# from . import query_iagos
from . import query_icos


CACHE_DIR = 'cache'

logger = logging.getLogger(__name__)


# for caching purposes
# TODO: do it properly
_stations = None
_variables = None


# mapping from standard ECV names to short variable names (used for time-line graphs)
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
    'Nitrous Oxide': 'N2O'
}


def _get_ri_query_module_by_ri(ris=None):
    if ris is None:
        ris = ['actris', 'iagos', 'icos']
    else:
        ris = sorted(ri.lower() for ri in ris)
    ri_query_module_by_ri = {}
    for ri in ris:
        if ri == 'actris':
            ri_query_module_by_ri[ri] = query_actris
        elif ri == 'iagos':
            pass
        elif ri == 'icos':
            ri_query_module_by_ri[ri] = query_icos
        else:
            raise ValueError(f'ri={ri}')
    return ri_query_module_by_ri


def _get_stations(ris=None):
    ri_query_module_by_ri = _get_ri_query_module_by_ri(ris)
    stations_dfs = []
    for ri, ri_query_module in ri_query_module_by_ri.items():
        cache_path = pathlib.PurePath(CACHE_DIR, f'stations_{ri}.pkl')
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
                    stations_dfs.append(stations_df)
            elif ri == 'iagos':
                raise NotImplementedError(ri)
            elif ri == 'icos':
                for col in ['latitude', 'longitude', 'ground_elevation']:
                    stations_df[col] = pd.to_numeric(stations_df[col])
                stations_dfs.append(stations_df)
            else:
                raise ValueError(f'ri={ri}')
        except Exception as e:
            logger.exception(f'getting {ri.upper()} stations failed', exc_info=e)

    all_stations_df = pd.concat(stations_dfs, ignore_index=True)
    all_stations_df['short_name_RI'] = all_stations_df['short_name'] + ' (' + all_stations_df['RI'] + ')'
    all_stations_df['idx'] = all_stations_df.index
    all_stations_df['marker_size'] = 7

    return all_stations_df


def get_stations():
    """
    For each ACTRIS, IAGOS and ICOS station (for the moment it is ICOS only).
    :return: pandas Dataframe with stations data; it has the following columns:
    'uri', 'short_name', 'long_name', 'country', 'latitude', 'longitude', 'ground_elevation', 'RI', 'short_name_RI',
    'theme', 'idx'
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
        'theme': 'AS'
        'idx': 2
    """
    global _stations
    if _stations is None:
        _stations = _get_stations(['actris', 'icos'])
    return _stations


def _get_vars_long():
    """
    Provide a listing of RI's variables. For the same variable code there might be many records with different ECV names
    :return: pandas.DataFrame with columns: 'variable_name', 'ECV_name'; sample records are:
        'variable_name': 'co', 'ECV_name': 'Carbon Monoxide', 'std_ECV_name': 'Carbon Monoxide';
        'variable_name': 'co', 'ECV_name': 'Carbon Dioxide, Methane and other Greenhouse gases', 'std_ECV_name': 'Carbon Monoxide';
        'variable_name': 'co', 'ECV_name': 'co', 'std_ECV_name': 'Carbon Monoxide';
    """
    global _variables
    if _variables is None:
        ri_query_module_by_ri = _get_ri_query_module_by_ri(['actris', 'icos'])
        variables_dfs = []
        for ri, ri_query_module in ri_query_module_by_ri.items():
            cache_path = pathlib.PurePath(CACHE_DIR, f'variables_{ri}.pkl')
            try:
                try:
                    variables_df = pd.read_pickle(cache_path)
                except FileNotFoundError:
                    variables = ri_query_module.get_list_variables()
                    variables_df = pd.DataFrame.from_dict(variables)
                    variables_df.to_pickle(cache_path)
                variables_dfs.append(variables_df)
            except Exception as e:
                logger.exception(f'getting {ri.upper()} variables failed', exc_info=e)
        df = pd.concat(variables_dfs, ignore_index=True)
        df['std_ECV_name'] = df['ECV_name'].apply(lambda l: l[0])
        df = df.join(pd.Series(VARIABLES_MAPPING, name='code'), on='std_ECV_name')
        _variables = df.explode('ECV_name', ignore_index=True)
    return _variables


def get_vars():
    """
    Provide a listing of RI's variables.
    :return: pandas.DataFrame with columns: 'variable_name', 'ECV_name'; a sample record is:
        'ECV_name': 'Carbon Dioxide, Methane and other Greenhouse gases',
        'std_ECV_name': 'Carbon Monoxide'
    """
    variables_df = _get_vars_long().drop(columns=['variable_name'])
    return variables_df.drop_duplicates(subset=['std_ECV_name', 'ECV_name'], keep='first', ignore_index=True)


def get_datasets(variables, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Provide metadata of datasets selected according to the provided criteria.
    :param variables: list of str or None; list of variable codes
    :param lon_min: float or None
    :param lon_max: float or None
    :param lat_min: float or None
    :param lat_max: float or None
    :return: pandas.DataFrame with columns: 'title', 'ecv_variables', 'platform_id', 'RI', 'platform_id_RI',
    'var_codes', 'var_codes_filtered', 'url', 'time_period_start', 'time_period_end';
    e.g. for the call get_datasets(['AP', 'AT'] one gets a dataframe with an example row like:
         'title': 'ICOS_ATC_L2_L2-2021.1_GAT_2.5_CTS_MTO.zip',
         'ecv_variables': ['Pressure (surface)', 'Surface Wind Speed and direction', 'Temperature (near surface)', 'Water Vapour (surface)'],
         'platform_id: 'GAT',
         'RI': 'ICOS',
         'platform_id_RI': 'GAT (ICOS)',
         'var_codes': ['AP', 'AT', 'RH', 'WD', 'WS'],
         'var_codes_filtered': 'AP,AT',
         'url':'https://meta.icos-cp.eu/objects/0HxLXMXolAVqfcuqpysYz8jK',
         'time_period_start': Timestamp('2016-05-10 00:00:00+0000', tz='UTC'),
         'time_period_end': Timestamp('2021-01-31 23:00:00+0000', tz='UTC')
    """
    if variables is None:
        variables = []
    if None in [lon_min, lon_max, lat_min, lat_max]:
        bbox = []
    else:
        bbox = [lon_min, lat_min, lon_max, lat_max]

    datasets_dfs = []
    for f in (_get_actris_datasets, _get_icos_datasets):
        df = f(variables, bbox)
        if df is not None:
            datasets_dfs.append(df)

    if not datasets_dfs:
        return None
    datasets_df = pd.concat(datasets_dfs, ignore_index=True)

    vars_long = _get_vars_long()
    def var_names_to_var_codes(var_names):
        # TODO: performance issues
        var_names = np.unique(var_names)
        codes1 = vars_long[['ECV_name', 'code']].join(pd.DataFrame(index=var_names), on='ECV_name', how='inner')
        codes1 = codes1['code'].unique()
        codes2 = vars_long[['variable_name', 'code']].join(pd.DataFrame(index=var_names), on='variable_name', how='inner')
        codes2 = codes2['code'].unique()
        codes = np.concatenate([codes1, codes2])
        return np.sort(np.unique(codes)).tolist()
    datasets_df['var_codes'] = datasets_df['ecv_variables'].apply(lambda var_names: var_names_to_var_codes(var_names))
    req_var_codes = set(VARIABLES_MAPPING[v] for v in variables if v in VARIABLES_MAPPING)
    datasets_df['var_codes_filtered'] = datasets_df['var_codes']\
        .apply(lambda var_codes: ', '.join([vc for vc in var_codes if vc in req_var_codes]))

    datasets_df['url'] = datasets_df['urls'].apply(lambda x: x[-1]['url'])  # now we take the last proposed url; TODO: see what should be a proper rule (first non-empty url?)
    datasets_df['time_period_start'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[0]))
    datasets_df['time_period_end'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[1]))
    datasets_df['platform_id_RI'] = datasets_df['platform_id'] + ' (' + datasets_df['RI'] + ')'

    return datasets_df.drop(columns=['urls', 'time_period'])


def _get_actris_datasets(variables, bbox):
    datasets = query_actris.query_datasets(variables=variables, temporal_extent=[], spatial_extent=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)
    datasets_df['RI'] = 'ACTRIS'
    return datasets_df


def _get_icos_datasets(variables, bbox):
    datasets = query_icos.query_datasets(variables=variables, temporal=[], spatial=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)
    datasets_df['RI'] = 'ICOS'
    return datasets_df


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
