import warnings
import numpy as np
import pandas as pd
import dash
from dash import Output, Input, State, Patch, callback

import data_access
import utils.stations_map
from app_tabs.common.data import stations
from app_tabs.common.layout import SELECTED_STATIONS_STORE_ID, SELECTED_ECV_STORE_ID, DATASETS_STORE_ID, APP_TABS_ID, \
    SELECT_DATASETS_TAB_VALUE
from app_tabs.search_datasets_tab.layout import VARIABLES_CHECKLIST_ID, VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, \
    SEARCH_DATASETS_BUTTON_ID, \
    SELECTED_STATIONS_DROPDOWN_ID, STATIONS_MAP_ID, SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID, \
    MAP_BACKGROUND_RADIO_ID, MAPBOX_STYLES, MAP_ZOOM_STORE_ID
from app_tabs.select_datasets_tab.layout import VARIABLES_LEGEND_DROPDOWN_ID, get_variables_legend_options
from utils.stations_map import DEFAULT_STATIONS_SIZE, SELECTED_STATIONS_OPACITY, UNSELECTED_STATIONS_OPACITY, \
    SELECTED_STATIONS_SIZE, UNSELECTED_STATIONS_SIZE
from data_processing.utils import points_inside_polygons
from utils.exception_handler import callback_with_exc_handling, AppException, AppWarning


@callback_with_exc_handling(
    Output(SELECTED_ECV_STORE_ID, 'data'),
    Input(VARIABLES_CHECKLIST_ID, 'value'),
    # prevent_initial_call=True,
)
def update_selected_ecv_store(selected_ecv):
    return selected_ecv


@callback_with_exc_handling(
    Output(VARIABLES_CHECKLIST_ID, 'value'),
    Input(VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    State(SELECTED_ECV_STORE_ID, 'data'),
    # prevent_initial_call=True,
)
def toogle_variable_checklist(variables_checklist_all_none_switch, selected_ecv):
    _all_variables = list(data_access.ECV_by_var_codes)

    ctx = list(dash.ctx.triggered_prop_ids.values())
    if VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID in ctx:
        if variables_checklist_all_none_switch:
            return _all_variables
        else:
            return []
    else:
        if selected_ecv is None:
            selected_ecv = _all_variables
        _res = [v for v in _all_variables if v in selected_ecv]
        # print(f'_res={_res}')
        return _res


@callback_with_exc_handling(
    Output(SEARCH_DATASETS_BUTTON_ID, 'disabled'),
    Input(SELECTED_STATIONS_STORE_ID, 'data'),
    Input(SELECTED_ECV_STORE_ID, 'data'),
)
def search_datasets_button_disabled(selected_stations_idx, selected_variables):
    return not (selected_stations_idx and selected_variables)


@callback_with_exc_handling(
    Output(STATIONS_MAP_ID, 'figure', allow_duplicate=True),
    Input(MAP_BACKGROUND_RADIO_ID, 'value'),
    prevent_initial_call=True
)
def change_map_background(map_background):
    if map_background not in MAPBOX_STYLES:
        raise dash.exceptions.PreventUpdate
    patched_fig = Patch()
    patched_fig['layout']['mapbox']['style'] = map_background
    return patched_fig


@callback_with_exc_handling(
    Output(VARIABLES_LEGEND_DROPDOWN_ID, 'value'),
    Output(VARIABLES_LEGEND_DROPDOWN_ID, 'options'),
    Input(SEARCH_DATASETS_BUTTON_ID, 'n_clicks'),
    State(SELECTED_ECV_STORE_ID, 'data')
)
def set_variables_legend_dropdown(n_click, selected_variables):
    return selected_variables, get_variables_legend_options(selected_variables)


@callback_with_exc_handling(
    Output(DATASETS_STORE_ID, 'data'),
    Output(APP_TABS_ID, 'active_tab', allow_duplicate=True),
    Input(SEARCH_DATASETS_BUTTON_ID, 'n_clicks'),
    State(SELECTED_ECV_STORE_ID, 'data'),
    State(SELECTED_STATIONS_STORE_ID, 'data'),
    prevent_initial_call=True
)
#@log_exectime
def search_datasets(n_clicks, selected_variables, selected_stations_idx):
    if not (selected_stations_idx and selected_variables):
        raise dash.exceptions.PreventUpdate

    empty_datasets_df = pd.DataFrame(
        columns=[
            'title', 'url', 'ecv_variables', 'platform_id', 'RI', 'var_codes', 'ecv_variables_filtered',
            'var_codes_filtered', 'time_period_start', 'time_period_end', 'platform_id_RI', 'id'
        ]
    )   # TODO: do it cleanly

    selected_stations = stations.iloc[selected_stations_idx]

    datasets_df = data_access.get_datasets(
        selected_variables,
        station_codes=selected_stations['short_name'],
        ris=selected_stations['RI']
    )

    if datasets_df is None:
        datasets_df = empty_datasets_df

    datasets_df_filtered = datasets_df[
        datasets_df['platform_id'].isin(selected_stations['short_name']) &
        datasets_df['RI'].isin(selected_stations['RI'])     # short_name of the station might not be unique among RI's
    ]

    datasets_df_filtered = datasets_df_filtered.reset_index(drop=True)
    datasets_df_filtered['id'] = datasets_df_filtered.index

    if len(datasets_df_filtered) > 0:
        new_tab = SELECT_DATASETS_TAB_VALUE
    else:
        new_tab = dash.no_update
        warnings.warn('No datasets found. Change search criteria.', category=AppWarning)

    return (
        datasets_df_filtered.to_json(orient='split', date_format='iso'),
        new_tab
    )


def _get_selected_points(selected_stations):
    if selected_stations is not None:
        points = selected_stations['points']
        for point in points:
            point['idx'] = round(point['customdata'][0])
    else:
        points = []
    return pd.DataFrame.from_records(points, index='idx', columns=['idx', 'lon', 'lat'])


def _get_selected_stations_dropdown(selected_stations_df, stations_dropdown_options):
    df = selected_stations_df
    labels = df['short_name'] + ' (' + df['long_name'] + ', ' + df['RI'] + ')'
    values = labels.index.to_list()
    options = labels.to_dict()
    if stations_dropdown_options is not None:
        existing_options = {opt['value']: opt['label'] for opt in stations_dropdown_options}
        options.update(existing_options)
    options = [{'value': value, 'label': label} for value, label in options.items()]
    return options, values
    # options = labels.rename('label').reset_index().rename(columns={'index': 'value'})
    # return options.to_dict(orient='records'), list(options['value'])


@callback_with_exc_handling(
    Output(SELECTED_STATIONS_STORE_ID, 'data'),
    Output(STATIONS_MAP_ID, 'figure'),
    Output(MAP_ZOOM_STORE_ID, 'data'),
    Output(SELECTED_STATIONS_DROPDOWN_ID, 'value'),
    Input(SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID, 'n_clicks'),
    Input(STATIONS_MAP_ID, 'selectedData'),
    Input(STATIONS_MAP_ID, 'clickData'),
    Input(STATIONS_MAP_ID, 'relayoutData'),
    Input(SELECTED_STATIONS_DROPDOWN_ID, 'value'),
    State(MAP_ZOOM_STORE_ID, 'data'),
    State(SELECTED_STATIONS_STORE_ID, 'data'),
    State(SELECTED_STATIONS_DROPDOWN_ID, 'options'),
)
def get_selected_stations_bbox_and_dropdown(
        reset_stations_button,
        selected_data,
        click_data,
        map_relayoutData,
        stations_dropdown,
        previous_zoom,
        selected_stations_idx,
        stations_dropdown_options
):
    # TODO: (refactor) move stations update to utils.stations_map module
    ctx_keys = list(dash.ctx.triggered_prop_ids)
    ctx_values = list(dash.ctx.triggered_prop_ids.values())

    # apply station unclustering
    zoom = map_relayoutData.get('mapbox.zoom', previous_zoom) if isinstance(map_relayoutData, dict) else previous_zoom
    # lon_3857_displaced, lat_3857_displaced, lon_displaced, lat_displaced = utils.stations_map.uncluster(stations['lon_3857'], stations['lat_3857'], zoom)

    if SELECTED_STATIONS_DROPDOWN_ID in ctx_values and stations_dropdown is not None:
        selected_stations_idx = sorted(stations_dropdown)

    if SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID in ctx_values:
        selected_stations_idx = None

    if (f'{STATIONS_MAP_ID}.selectedData' in ctx_keys or f'{STATIONS_MAP_ID}.clickData' in ctx_keys)\
            and selected_data and 'points' in selected_data and len(selected_data['points']) > 0:
        if 'range' in selected_data or 'lassoPoints' in selected_data:
            currently_selected_stations_on_map_idx = [
                p['customdata'][0]
                for p in selected_data['points']
                if 'customdata' in p  # take account only traces corresponding to stations / regions
                                      # and NOT to displacement vectors or original stations' positions
            ]
            # it does not contain points from categories which were deselected on the legend
            # however might contain points that we clicked on while keeping the key 'shift' pressed
            # some of the latter might have been actually un-clicked (deselected)
            # that's why we compute stations_in_current_selection_idx (indices of stations inside the current range
            # or lasso selection, regardless their category is switched on or off)
            if 'range' in selected_data and 'mapbox' in selected_data['range']:
                mapbox_range = selected_data['range']['mapbox']
                try:
                    (lon1, lat1), (lon2, lat2) = mapbox_range
                except TypeError:
                    print(f'mapbox_range: {mapbox_range}')
                    raise dash.exceptions.PreventUpdate
                lon1, lon2 = sorted([lon1, lon2])
                lat1, lat2 = sorted([lat1, lat2])
                lon = stations['longitude']
                lat = stations['latitude']
                stations_in_current_selection_idx = stations[(lon >= lon1) & (lon <= lon2) & (lat >= lat1) & (lat <= lat2)].index
            elif 'lassoPoints' in selected_data and 'mapbox' in selected_data['lassoPoints']:
                mapbox_lassoPoints = selected_data['lassoPoints']['mapbox']
                mapbox_lassoPoints = np.array(mapbox_lassoPoints).T
                points = np.array([stations['longitude'].values, stations['latitude'].values])
                stations_in_current_selection_idx, = np.nonzero(points_inside_polygons(points, mapbox_lassoPoints))
            else:
                stations_in_current_selection_idx = []
            stations_in_current_selection_idx = set(stations_in_current_selection_idx).intersection(currently_selected_stations_on_map_idx)
            # we take the intersection because some categories might be switched off on the legend

            selected_stations_idx = sorted(set(selected_stations_idx if selected_stations_idx is not None else []).union(stations_in_current_selection_idx))
        elif click_data and 'points' in click_data:
            currently_selected_stations_on_map_idx = [p['customdata'][0] for p in click_data['points']]
            selected_stations_idx = sorted(set(selected_stations_idx if selected_stations_idx is not None else []).symmetric_difference(currently_selected_stations_on_map_idx))

    size = pd.Series(UNSELECTED_STATIONS_SIZE, index=stations.index)
    opacity = pd.Series(UNSELECTED_STATIONS_OPACITY, index=stations.index)

    # this change is OK since we are going to delete bounding box feature
    # if selected_stations_idx is not None:
    if selected_stations_idx:
        opacity.loc[selected_stations_idx] = SELECTED_STATIONS_OPACITY
        size.loc[selected_stations_idx] = SELECTED_STATIONS_SIZE
    else:
        opacity.loc[:] = SELECTED_STATIONS_OPACITY
        size.loc[:] = DEFAULT_STATIONS_SIZE

    patched_fig = utils.stations_map.get_stations_map_patch(zoom, size, opacity)

    selected_stations_df = stations.loc[selected_stations_idx] if selected_stations_idx is not None else stations.loc[[]]

    return selected_stations_idx, patched_fig, zoom, list(selected_stations_df.index)
