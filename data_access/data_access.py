"""
This module provides a unified API for access to metadata and datasets from ACTRIS, IAGOS and ICOS RI's.
At the moment, the access to ICOS RI is implemented here.
"""

import numpy as np
import pandas as pd

# from . import query_actris
# from . import query_iagos
from . import query_icos


# for caching purposes
# TODO: do it properly
_stations = None
_variables = None


def get_stations():
    """
    For each ACTRIS, IAGOS and ICOS station (for the moment it is ICOS only).
    :return: pandas Dataframe with stations data; it has the following columns:
    'uri', 'short_name', 'long_name', 'country', 'latitude', 'longitude', 'ground_elevation', 'RI', 'theme'.
    A sample record is:
        'uri': 'http://meta.icos-cp.eu/resources/stations/AS_BIR',
        'short_name': 'BIR',
        'long_name': 'Birkenes',
        'country': 'NO',
        'latitude': 58.3886,
        'longitude': 8.2519,
        'ground_elevation': 219.0,
        'RI': 'ICOS',
        'theme': 'AS'
    """
    global _stations
    if _stations is None:
        icos_stations = query_icos.get_list_platforms()
        icos_stations_df = pd.DataFrame.from_dict(icos_stations)
        for col in ['latitude', 'longitude', 'ground_elevation']:
            icos_stations_df[col] = pd.to_numeric(icos_stations_df[col])
        _stations = icos_stations_df
    return _stations


def _get_vars_long():
    """
    Provide a listing of RI's variables. For the same variable code there might be many records with different ECV names
    :return: pandas.DataFrame with columns: 'variable_name', 'ECV_name'; sample records are:
        'variable_name': 'co', 'ECV_name': 'Carbon Monoxide';
        'variable_name': 'co', 'ECV_name': 'Carbon Dioxide, Methane and other Greenhouse gases';
        'variable_name': 'co', 'ECV_name': 'co';
    """
    global _variables
    if _variables is None:
        variables = query_icos.get_list_variables()
        variables_df = pd.DataFrame.from_dict(variables)
        _variables = variables_df.explode('ECV_name', ignore_index=True)
    return _variables


def get_vars():
    """
    Provide a listing of RI's variables.
    :return: pandas.DataFrame with columns: 'variable_name', 'ECV_name'; a sample record is:
        'variable_name': 'co',
        'ECV_name': 'Carbon Monoxide'
    """
    variables_df = _get_vars_long()
    return variables_df.drop_duplicates(subset='variable_name', keep='first', ignore_index=True)


def get_datasets(variables, lon_min=None, lon_max=None, lat_min=None, lat_max=None):
    """
    Provide metadata of datasets selected according to the provided criteria.
    :param variables: list of str or None; list of variable codes
    :param lon_min: float or None
    :param lon_max: float or None
    :param lat_min: float or None
    :param lat_max: float or None
    :return: pandas.DataFrame with columns: 'title', 'ecv_variables', 'platform_id', 'var_codes', 'var_codes_filtered',
    'url', 'time_period_start', 'time_period_end';
    e.g. for the call get_datasets(['AP', 'AT'] one gets a dataframe with an example row like:
         'title': 'ICOS_ATC_L2_L2-2021.1_GAT_2.5_CTS_MTO.zip',
         'ecv_variables': ['Pressure (surface)', 'Surface Wind Speed and direction', 'Temperature (near surface)', 'Water Vapour (surface)'],
         'platform_id: 'GAT',
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
    datasets = query_icos.query_datasets(variables=variables, temporal=[], spatial=bbox)
    if not datasets:
        return None
    datasets_df = pd.DataFrame.from_dict(datasets)
    vars_long = _get_vars_long()

    def var_names_to_var_codes(var_names):
        df = vars_long.join(pd.DataFrame(index=var_names), on='ECV_name', how='inner')
        return np.sort(np.unique(df['variable_name'])).tolist()

    datasets_df['var_codes'] = datasets_df['ecv_variables'].apply(lambda var_names: var_names_to_var_codes(var_names))
    variables = set(variables)
    datasets_df['var_codes_filtered'] = datasets_df['var_codes'].apply(lambda var_codes: ','.join([vc for vc in var_codes if vc in variables]))
    datasets_df['url'] = datasets_df['urls'].apply(lambda x: x[0]['url'])
    datasets_df['time_period_start'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[0]))
    datasets_df['time_period_end'] = datasets_df['time_period'].apply(lambda x: pd.Timestamp(x[1]))
    return datasets_df.drop(columns=['urls', 'time_period'])


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
