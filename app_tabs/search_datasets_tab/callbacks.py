import pathlib

import numpy as np
import pandas as pd
import pkg_resources
from dash import Output, Input, State, callback

import data_access
import data_access.common
from app_tabs.common.data import stations
from app_tabs.common.layout import DATASETS_STORE_ID
from app_tabs.search_datasets_tab.layout import VARIABLES_CHECKLIST_ID, VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, \
    std_variables, SEARCH_DATASETS_BUTTON_ID, LON_MIN_ID, LON_MAX_ID, LAT_MIN_ID, LAT_MAX_ID, \
    SELECTED_STATIONS_DROPDOWN_ID, STATIONS_MAP_ID
from log import logger, log_exception


DEBUG_GET_DATASETS = False


@callback(
    Output(VARIABLES_CHECKLIST_ID, 'value'),
    Input(VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value')
)
@log_exception
def toogle_variable_checklist(variables_checklist_all_none_switch):
    if variables_checklist_all_none_switch:
        return std_variables['value'].tolist()
    else:
        return []


@callback(
    Output(DATASETS_STORE_ID, 'data'),
    Input(SEARCH_DATASETS_BUTTON_ID, 'n_clicks'),
    State(VARIABLES_CHECKLIST_ID, 'value'),
    State(LON_MIN_ID, 'value'),
    State(LON_MAX_ID, 'value'),
    State(LAT_MIN_ID, 'value'),
    State(LAT_MAX_ID, 'value'),
    State(SELECTED_STATIONS_DROPDOWN_ID, 'value'),
    State(DATASETS_STORE_ID, 'data'),  # TODO: if no station or variable selected, do not launch Search datasets action; instead, return an old data
)
@log_exception
def search_datasets(
        n_clicks, selected_variables, lon_min, lon_max, lat_min, lat_max,
        selected_stations_idx, previous_datasets_json
):
    if selected_stations_idx is None:
        selected_stations_idx = []

    empty_datasets_df = pd.DataFrame(
        columns=['title', 'url', 'ecv_variables', 'platform_id', 'RI', 'var_codes', 'ecv_variables_filtered',
                 'std_ecv_variables_filtered', 'var_codes_filtered', 'time_period_start', 'time_period_end',
                 'platform_id_RI', 'id']
    )   # TODO: do it cleanly

    if not selected_variables or None in [lon_min, lon_max, lat_min, lat_max]:
        if previous_datasets_json is not None:
            datasets_json = previous_datasets_json
        else:
            datasets_json = empty_datasets_df.to_json(orient='split', date_format='iso')
        return datasets_json

    datasets_df = data_access.get_datasets(selected_variables, lon_min, lon_max, lat_min, lat_max)
    if DEBUG_GET_DATASETS:
        datasets_df2 = data_access.get_datasets_old(selected_variables, lon_min, lon_max, lat_min, lat_max)
        datasets_df_not_match = False
        if datasets_df is None and datasets_df2 is not None or datasets_df is not None and datasets_df2 is None:
            datasets_df_not_match = True
        elif datasets_df is not None:
            datasets_df_not_match = datasets_df.equals(datasets_df2)
        if not datasets_df_not_match:
            logger().error(f'datasets dfs do not match: selected_variables={selected_variables}, '
                           f'lon_min={lon_min}, lon_max={lon_max}, lat_min={lat_min}, lat_max={lat_max}\n'
                           f'datasets_df={datasets_df}\n'
                           f'datasets_df2={datasets_df2}')
            datasets_df.to_pickle(data_access.common.CACHE_DIR / '_datasets_df.pkl')
            datasets_df2.to_pickle(data_access.common.CACHE_DIR / '_datasets_df2.pkl')
        else:
            logger().info('datasets_df == datasets_df2')

    if datasets_df is None:
        datasets_df = empty_datasets_df

    selected_stations = stations.iloc[selected_stations_idx]
    datasets_df_filtered = datasets_df[
        datasets_df['platform_id'].isin(selected_stations['short_name']) &
        datasets_df['RI'].isin(selected_stations['RI'])     # short_name of the station might not be unique among RI's
    ]

    datasets_df_filtered = datasets_df_filtered.reset_index(drop=True)
    datasets_df_filtered['id'] = datasets_df_filtered.index

    return datasets_df_filtered.to_json(orient='split', date_format='iso')


def _get_selected_points(selected_stations):
    if selected_stations is not None:
        points = selected_stations['points']
        for point in points:
            point['idx'] = round(point['customdata'][0])
    else:
        points = []
    return pd.DataFrame.from_records(points, index='idx', columns=['idx', 'lon', 'lat'])


def _get_bounding_box(selected_points_df, selected_stations):
    # decimal precision for bounding box coordinates (lon/lat)
    decimal_precision = 2

    # find selection box, if there is one
    try:
        (lon_min, lat_max), (lon_max, lat_min) = selected_stations['range']['mapbox']
    except:
        lon_min, lon_max, lat_min, lat_max = np.inf, -np.inf, np.inf, -np.inf

    if len(selected_points_df) > 0:
        # find bouding box for selected points
        epsilon = 0.001  # precision margin for filtering on lon/lat of stations later on
        lon_min2, lon_max2 = selected_points_df['lon'].min() - epsilon, selected_points_df['lon'].max() + epsilon
        lat_min2, lat_max2 = selected_points_df['lat'].min() - epsilon, selected_points_df['lat'].max() + epsilon

        # find a common bounding box for the both bboxes found above
        lon_min, lon_max = np.min((lon_min, lon_min2)), np.max((lon_max, lon_max2))
        lat_min, lat_max = np.min((lat_min, lat_min2)), np.max((lat_max, lat_max2))

    if not np.all(np.isfinite([lon_min, lon_max, lat_min, lat_max])):
        return [None] * 4
    return [round(coord, decimal_precision) for coord in (lon_min, lon_max, lat_min, lat_max)]


def _get_selected_stations_dropdown(selected_stations_df):
    idx = selected_stations_df.index
    df = stations.iloc[idx]
    labels = df['short_name'] + ' (' + df['long_name'] + ', ' + df['RI'] + ')'
    options = labels.rename('label').reset_index().rename(columns={'index': 'value'})
    return options.to_dict(orient='records'), list(options['value'])


@callback(
    Output(LON_MIN_ID, 'value'),
    Output(LON_MAX_ID, 'value'),
    Output(LAT_MIN_ID, 'value'),
    Output(LAT_MAX_ID, 'value'),
    Output(SELECTED_STATIONS_DROPDOWN_ID, 'options'),
    Output(SELECTED_STATIONS_DROPDOWN_ID, 'value'),
    Input(STATIONS_MAP_ID, 'selectedData'))
@log_exception
def get_selected_stations_bbox_and_dropdown(selected_stations):
    selected_stations_df = _get_selected_points(selected_stations)
    bbox = _get_bounding_box(selected_stations_df, selected_stations)
    selected_stations_dropdown_options, selected_stations_dropdown_value = _get_selected_stations_dropdown(selected_stations_df)
    return bbox + [selected_stations_dropdown_options, selected_stations_dropdown_value]
