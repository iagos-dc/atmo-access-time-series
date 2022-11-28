"""
ATMO-ACCESS time series service
"""

import pathlib
import pkg_resources
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import json
import werkzeug.utils

# import gunicorn

# Dash imports; for documentation (including tutorial), see: https://dash.plotly.com/
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc

# Provides a version of Dash application which can be run in Jupyter notebook/lab
# See: https://github.com/plotly/jupyter-dash
from dash import Dash

# Local imports
from log import logger
import data_access
import data_processing
from utils.charts import ACTRIS_COLOR_HEX, IAGOS_COLOR_HEX, ICOS_COLOR_HEX, _get_scatter_plot, _get_line_plot, \
    _get_timeline_by_station, _get_timeline_by_station_and_vars, _plot_vars, get_avail_data_by_var_gantt

CACHE_DIR = pathlib.PurePath(pkg_resources.resource_filename('data_access', 'cache'))
DEBUG_GET_DATASETS = False


# Configuration of the app
# See: https://dash.plotly.com/devtools#configuring-with-run_server
# for the usual Dash app, and:
# https://github.com/plotly/jupyter-dash/blob/master/notebooks/getting_started.ipynb
# for a JupyterDash app version.
# app_conf = {'mode': 'external', 'debug': True}  # for running inside a Jupyter notebook change 'mode' to 'inline'
# RUNNING_IN_BINDER = os.environ.get('BINDER_SERVICE_HOST') is not None
# if RUNNING_IN_BINDER:
#     JupyterDash.infer_jupyter_proxy_config()
# else:
#     app_conf.update({'host': 'localhost', 'port': 9235})


# Below there are id's of Dash JS components.
# The components themselves are declared in the dashboard layout (see the function get_dashboard_layout).
# Essential properties of each component are explained in the comments below.
APP_TABS_ID = 'app-tabs'    # see: https://dash.plotly.com/dash-core-components/tabs; method 1 (content as callback)
    # value contains an id of the active tab
    # children contains a list of layouts of each tab
SEARCH_DATASETS_TAB_VALUE = 'search-datasets-tab'
SELECT_DATASETS_TAB_VALUE = 'select-datasets-tab'
FILTER_DATA_TAB_VALUE = 'filter-data-tab'
DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'

STATIONS_MAP_ID = 'stations-map'
    # 'selectedData' contains a dictionary
    # {
    #   'point' ->
    #       list of dicionaries {'pointIndex' -> index of a station in the global dataframe stations, 'lon' -> float, 'lat' -> float, ...},
    #   'range' (present only if a box was selected on the map) ->
    #       {'mapbox' -> [[lon_min, lat_max], [lon_max, lat_min]]}
    # }
VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'variables-checklist-all-none-switch'
VARIABLES_CHECKLIST_ID = 'variables-checklist'
SELECTED_STATIONS_DROPDOWN_ID = 'selected-stations-dropdown'
    # 'options' contains a list of dictionaries {'label' -> station label, 'value' -> index of the station in the global dataframe stations (see below)}
    # 'value' contains a list of indices of stations selected using the dropdown
SEARCH_DATASETS_BUTTON_ID = 'search-datasets-button'
    # 'n_click' contains a number of clicks at the button
LAT_MAX_ID = 'lat-max'
LAT_MIN_ID = 'lat-min'
LON_MAX_ID = 'lon-max'
LON_MIN_ID = 'lon-min'
    # 'value' contains a number (or None)
GANTT_VIEW_RADIO_ID = 'gantt-view-radio'
    # 'value' contains 'compact' or 'detailed'
GANTT_GRAPH_ID = 'gantt-graph'
    # 'figure' contains a Plotly figure object
DATASETS_STORE_ID = 'datasets-store'
    # 'data' stores datasets metadata in JSON, as provided by the method pd.DataFrame.to_json(orient='split', date_format='iso')
SELECT_DATASETS_REQUEST_ID = 'select-datasets-request'
    # 'data' stores a JSON representation of a request executed
DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID = 'datasets-table-checklist-all-none-switch'
DATASETS_TABLE_ID = 'datasets-table'
    # 'columns' contains list of dictionaries {'name' -> column name, 'id' -> column id}
    # 'data' contains a list of records as provided by the method pd.DataFrame.to_dict(orient='records')
QUICKLOOK_POPUP_ID = 'quicklook-popup'
    # 'children' contains a layout of the popup
SELECT_DATASETS_BUTTON_ID = 'select-datasets-button'
    # 'n_click' contains a number of clicks at the button
FILTER_TAB_CONTAINER_ID = 'filter-tab-container'
    # 'children' contains a layout of the filter tab
# AVAIL_DATA_GRAPH_ID = 'avail-data-graph'
    # 'figure' contains a Plotly figure object
# VAR_HIST_GRAPHS_CONTAINER_ID = 'var-hist-graphs-container'
# SCATTER_GRAPH_ID = 'scatter-graph'
    # 'figure' contains a Plotly figure object

# Atmo-Access logo url
ATMO_ACCESS_LOGO_URL = \
    'https://www7.obs-mip.fr/wp-content-aeris/uploads/sites/82/2021/03/ATMO-ACCESS-Logo-final_horizontal-payoff-grey-blue.png'


def _get_station_by_shortnameRI(stations):
    df = stations.set_index('short_name_RI')[['long_name', 'RI']]
    df['station_fullname'] = df['long_name'] + ' (' + df['RI'] + ')'
    return df


def _get_std_variables(variables):
    std_vars = variables[['std_ECV_name', 'code']].drop_duplicates()
    # TODO: temporary
    try:
        std_vars = std_vars[std_vars['std_ECV_name'] != 'Aerosol Optical Properties']
    except ValueError:
        pass
    std_vars['label'] = std_vars['code'] + ' - ' + std_vars['std_ECV_name']
    return std_vars.rename(columns={'std_ECV_name': 'value'}).drop(columns='code')


# Initialization of global objects
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
)
server = app.server

stations = data_access.get_stations()
station_by_shortnameRI = _get_station_by_shortnameRI(stations)
variables = data_access.get_vars()
std_variables = _get_std_variables(variables)


# Begin of definition of routines which constructs components of the dashboard

def get_variables_checklist():
    """
    Provide variables checklist Dash component
    See: https://dash.plotly.com/dash-core-components/checklist
    :return: dash.dcc.Checklist
    """
    variables_options = std_variables.to_dict(orient='records')
    variables_values = std_variables['value'].tolist()
    variables_checklist = dbc.Checklist(
        id=VARIABLES_CHECKLIST_ID,
        options=variables_options,
        value=variables_values,
        labelStyle={'display': 'flex'},  # display in column rather than in a row; not sure if it is the right way to do
    )
    return variables_checklist


# TODO: if want to move to utils.charts, need to take care about external variables: stations and STATIONS_MAP_ID
def get_stations_map():
    """
    Provide a Dash component containing a map with stations
    See: https://dash.plotly.com/dash-core-components/graph
    :return: dash.dcc.Graph object
    """
    fig = px.scatter_mapbox(
        stations,
        lat="latitude", lon="longitude", color='RI',
        hover_name="long_name",
        hover_data={
            'RI': True,
            'longitude': ':.2f',
            'latitude': ':.2f',
            'ground elevation': stations['ground_elevation'].round(0).fillna('N/A').to_list(),
            'marker_size': False
        },
        custom_data=['idx'],
        size=stations['marker_size'],
        size_max=7,
        category_orders={'RI': ['ACTRIS', 'IAGOS', 'ICOS']},
        color_discrete_sequence=[ACTRIS_COLOR_HEX, IAGOS_COLOR_HEX, ICOS_COLOR_HEX],
        zoom=2,
        # width=1200, height=700,
        center={'lon': 10, 'lat': 55},
        title='Stations map',
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={'autoexpand': True, 'r': 0, 't': 40, 'l': 0, 'b': 0},
        # width=1100, height=700,
        autosize=True,
        clickmode='event+select',
        dragmode='select',
        hoverdistance=1, hovermode='closest',  # hoverlabel=None,
    )

    regions = stations[stations['is_region']]
    regions_lon = []
    regions_lat = []
    for lon_min, lon_max, lat_min, lat_max in zip(regions['longitude_min'], regions['longitude_max'], regions['latitude_min'], regions['latitude_max']):
        if len(regions_lon) > 0:
            regions_lon.append(None)
            regions_lat.append(None)
        regions_lon.extend([lon_min, lon_min, lon_max, lon_max, lon_min])
        regions_lat.extend([lat_min, lat_max, lat_max, lat_min, lat_min])

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        fill="toself",
        fillcolor='rgba(69, 96, 150, 0.05)',  # IAGOS_COLOR_HEX as rgba with opacity=0.05
        lon=regions_lon,
        lat=regions_lat,
        marker={'color': IAGOS_COLOR_HEX},
        name='IAGOS',
        legendgroup='IAGOS',
        opacity=0.7
    ))

    # TODO: synchronize box selection on the map with max/min lon/lat input fields
    # TODO: as explained in https://dash.plotly.com/interactive-graphing (Generic Crossfilter Recipe)
    stations_map = dcc.Graph(
        id=STATIONS_MAP_ID,
        figure=fig,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'scrollZoom': True,
        }
    )
    return stations_map



def get_bbox_selection_div():
    """
    Provide a composed Dash component with input/ouput text fields which allow to provide coordinates of a bounding box
    See: https://dash.plotly.com/dash-core-components/input
    :return: dash.html.Div object
    """
    bbox_selection_div = html.Div(id='bbox-selection-div', style={'margin-top': '15px'}, children=[
        html.Div(className='row', children=[
            html.Div(className='three columns, offset-by-six columns', children=[
                dcc.Input(id=LAT_MAX_ID, style={'width': '120%'}, placeholder='lat max', type='number', min=-90, max=90),  # , step=0.01),
            ]),
        ]),
        html.Div(className='row', children=[
            html.Div(className='three columns',
                     children=html.P(children='Bounding box:', style={'width': '100%', 'font-weight': 'bold'})),
            html.Div(className='three columns',
                     children=dcc.Input(style={'width': '120%'}, id=LON_MIN_ID, placeholder='lon min', type='number',
                                        min=-180, max=180),  # , step=0.01),
                     ),
            html.Div(className='offset-by-three columns, three columns',
                     children=dcc.Input(style={'width': '120%'}, id=LON_MAX_ID, placeholder='lon max', type='number',
                                        min=-180, max=180),  # , step=0.01),
                     ),
        ]),
        html.Div(className='row', children=[
            html.Div(className='offset-by-six columns, three columns',
                     children=dcc.Input(style={'width': '120%'}, id=LAT_MIN_ID, placeholder='lat min', type='number',
                                        min=-90, max=90),  # , step=0.01),
                     ),
        ]),
    ])
    return bbox_selection_div


def get_dashboard_layout():
    # these are special Dash components used for transferring data from one callback to other callback(s)
    # without displaying the data
    stores = [
        dcc.Store(id=DATASETS_STORE_ID, storage_type='session'),
        dcc.Store(id=SELECT_DATASETS_REQUEST_ID, storage_type='session'),
    ]

    # logo and application title
    title_and_logo_bar = html.Div(style={'display': 'flex', 'justify-content': 'space-between',
                                         'margin-bottom': '20px'},
                                  children=[
        html.Div(children=[
            html.H2('Time-series analysis', style={'font-weight': 'bold'}),
        ]),
        html.Div(children=[
            html.A(
                html.Img(
                    #src=app.get_asset_url('atmo_access_logo.png') if not RUNNING_IN_BINDER else ATMO_ACCESS_LOGO_URL,
                    src=ATMO_ACCESS_LOGO_URL,
                    style={'float': 'right', 'height': '70px', 'margin-top': '10px'}
                ),
                href="https://www.atmo-access.eu/",
            ),
        ]),
    ])

    stations_vars_tab = dcc.Tab(label='Search datasets', value=SEARCH_DATASETS_TAB_VALUE,
                                children=html.Div(style={'margin': '20px'}, children=[
        html.Div(id='search-datasets-left-panel-div', className='four columns', children=[
            html.Div(id='variables-selection-div', className='nine columns', children=[
                html.P('Select variable(s):', style={'font-weight': 'bold'}),
                dbc.Switch(
                    id=VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID,
                    label='Select all / none',
                    style={'margin-top': '10px'},
                    value=True,
                ),
                get_variables_checklist(),
            ]),

            html.Div(id='search-datasets-button-div', className='three columns',
                     children=dbc.Button(id=SEARCH_DATASETS_BUTTON_ID, n_clicks=0,
                                         color='primary',
                                         type='submit',
                                         style={'font-weight': 'bold'},
                                         children='Search datasets')),

            html.Div(id='search-datasets-left-panel-cont-div', className='twelve columns',
                     style={'margin-top': '20px'},
                     children=[
                         html.Div(children=[
                             html.P('Date range:', style={'display': 'inline', 'font-weight': 'bold', 'margin-right': '20px'}),
                             dcc.DatePickerRange(
                                 id='my-date-picker-range',
                                 min_date_allowed=datetime.date(1900, 1, 1),
                                 max_date_allowed=datetime.date(2022, 12, 31),
                                 initial_visible_month=datetime.date(2017, 8, 5),
                                 end_date=datetime.date(2017, 8, 25)
                             ),
                         ]),
                         get_bbox_selection_div(),
                     ]),
        ]),

        html.Div(id='search-datasets-right-panel-div', className='eight columns', children=[
            get_stations_map(),

            html.Div(id='selected-stations-div',
                     style={'margin-top': '20px'},
                     children=[
                         html.P('Selected stations (you can refine your selection)',
                                style={'font-weight': 'bold'}),
                         dcc.Dropdown(id=SELECTED_STATIONS_DROPDOWN_ID, multi=True,
                                      clearable=False),
            ]),
        ]),
    ]))

    select_datasets_tab = dcc.Tab(label='Select datasets', value=SELECT_DATASETS_TAB_VALUE,
                                  children=html.Div(style={'margin': '20px'}, children=[
        html.Div(id='select-datasets-left-panel-div', className='four columns', children=[
            html.Div(id='select-datasets-left-left-subpanel-div', className='nine columns', children=
                dbc.RadioItems(
                    id=GANTT_VIEW_RADIO_ID,
                    options=[
                        {'label': 'compact view', 'value': 'compact'},
                        {'label': 'detailed view', 'value': 'detailed'},
                    ],
                    value='compact',
                    inline=True)),
            html.Div(id='select-datasets-left-right-subpanel-div', className='three columns', children=
                dbc.Button(id=SELECT_DATASETS_BUTTON_ID, n_clicks=0,
                       color='primary', type='submit',
                       style={'font-weight': 'bold'},
                       children='Select datasets'))
        ]),
        html.Div(id='select-datasets-right-panel-div', className='eight columns', children=None),
        html.Div(id='select-datasets-main-panel-div', className='twelve columns', children=[
            dcc.Graph(
                id=GANTT_GRAPH_ID,
            ),
            dbc.Switch(
                id=DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID,
                label='Select all / none',
                style={'margin-top': '10px'},
                value=False,
            ),
            dash_table.DataTable(
                id=DATASETS_TABLE_ID,
                css=[dict(selector="p", rule="margin: 0px;")],
                # see: https://dash.plotly.com/datatable/interactivity
                row_selectable="multi",
                selected_rows=[],
                selected_row_ids=[],
                sort_action='native',
                # filter_action='native',
                page_action="native", page_current=0, page_size=20,
                # see: https://dash.plotly.com/datatable/width
                # hidden_columns=['url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'],
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'lineHeight': '15px'
                },
                style_cell={'textAlign': 'left'},
                markdown_options={'html': True},
            ),
            html.Div(id=QUICKLOOK_POPUP_ID),
        ]),
    ]))

    filter_data_tab = dcc.Tab(
        label='Filter data',
        value=FILTER_DATA_TAB_VALUE,
        children=html.Div(
            style={'margin': '20px'},
            children=dbc.Container(id=FILTER_TAB_CONTAINER_ID, fluid=True)
        )
    )

    mockup_remaining_tabs = _get_mockup_remaining_tabs()

    app_tabs = dcc.Tabs(id=APP_TABS_ID, value=SEARCH_DATASETS_TAB_VALUE,
                        children=[
                            stations_vars_tab,
                            select_datasets_tab,
                            filter_data_tab,
                            *mockup_remaining_tabs
                        ])

    layout = html.Div(id='app-container-div', style={'margin': '30px', 'padding-bottom': '50px'}, children=stores + [
        html.Div(id='heading-div', className='twelve columns', children=[
            title_and_logo_bar,
            app_tabs,
        ])
    ])

    return layout


def _get_mockup_remaining_tabs():
    data_analysis_tab = dcc.Tab(label='Data analysis', value=DATA_ANALYSIS_TAB_VALUE)
    return [data_analysis_tab]

# End of definition of routines which constructs components of the dashboard


# Assign a dashboard layout to app Dash object
app.layout = get_dashboard_layout()


# Begin of callback definitions and their helper routines.
# See: https://dash.plotly.com/basic-callbacks
# for a basic tutorial and
# https://dash.plotly.com/  -->  Dash Callback in left menu
# for more detailed documentation

@app.callback(
    Output(VARIABLES_CHECKLIST_ID, 'value'),
    Input(VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value')
)
def toogle_variable_checklist(variables_checklist_all_none_switch):
    if variables_checklist_all_none_switch:
        return std_variables['value'].tolist()
    else:
        return []


@app.callback(
    Output(APP_TABS_ID, 'value'),
    Input(SEARCH_DATASETS_BUTTON_ID, 'n_clicks'),
    Input(SELECT_DATASETS_BUTTON_ID, 'n_clicks'),
)
def change_app_tab(search_datasets_button_clicks, select_datasets_button_clicks):
    # trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    trigger = dash.ctx.triggered_id
    if trigger == SEARCH_DATASETS_BUTTON_ID:
        return SELECT_DATASETS_TAB_VALUE
    elif trigger == SELECT_DATASETS_BUTTON_ID:
        return FILTER_DATA_TAB_VALUE
    else:
        return SEARCH_DATASETS_TAB_VALUE


@app.callback(
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
            datasets_df.to_pickle(CACHE_DIR / '_datasets_df.pkl')
            datasets_df2.to_pickle(CACHE_DIR / '_datasets_df2.pkl')
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


@app.callback(
    Output(LON_MIN_ID, 'value'),
    Output(LON_MAX_ID, 'value'),
    Output(LAT_MIN_ID, 'value'),
    Output(LAT_MAX_ID, 'value'),
    Output(SELECTED_STATIONS_DROPDOWN_ID, 'options'),
    Output(SELECTED_STATIONS_DROPDOWN_ID, 'value'),
    Input(STATIONS_MAP_ID, 'selectedData'))
def get_selected_stations_bbox_and_dropdown(selected_stations):
    selected_stations_df = _get_selected_points(selected_stations)
    bbox = _get_bounding_box(selected_stations_df, selected_stations)
    selected_stations_dropdown_options, selected_stations_dropdown_value = _get_selected_stations_dropdown(selected_stations_df)
    return bbox + [selected_stations_dropdown_options, selected_stations_dropdown_value]


@app.callback(
    Output(GANTT_GRAPH_ID, 'figure'),
    Output(GANTT_GRAPH_ID, 'selectedData'),
    Input(GANTT_VIEW_RADIO_ID, 'value'),
    Input(DATASETS_STORE_ID, 'data'),
    prevent_initial_call=True,
)
def get_gantt_figure(gantt_view_type, datasets_json):
    selectedData = {'points': []}

    if datasets_json is None:
       return {}, selectedData   # empty figure; TODO: is it a right way?

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    datasets_df = datasets_df.join(station_by_shortnameRI['station_fullname'], on='platform_id_RI')  # column 'station_fullname' joined to datasets_df

    if len(datasets_df) == 0:
       return {}, selectedData   # empty figure; TODO: is it a right way?

    if gantt_view_type == 'compact':
        fig = _get_timeline_by_station(datasets_df)
    else:
        fig = _get_timeline_by_station_and_vars(datasets_df)
    fig.update_traces(
        selectedpoints=[],
        #mode='markers+text', marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 20},
        unselected={'marker': {'opacity': 0.4}, }
    )
    return fig, selectedData


@app.callback(
    Output(DATASETS_TABLE_ID, 'columns'),
    Output(DATASETS_TABLE_ID, 'data'),
    Output(DATASETS_TABLE_ID, 'selected_rows'),
    Output(DATASETS_TABLE_ID, 'selected_row_ids'),
    Input(GANTT_GRAPH_ID, 'selectedData'),
    Input(DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    State(DATASETS_STORE_ID, 'data'),
    State(DATASETS_TABLE_ID, 'selected_row_ids'),
    prevent_initial_call=True,
)
def datasets_as_table(gantt_figure_selectedData, datasets_table_checklist_all_none_switch,
                      datasets_json, previously_selected_row_ids):
    table_col_ids = ['eye', 'title', 'var_codes_filtered', 'RI', 'long_name', 'platform_id', 'time_period_start', 'time_period_end',
                     #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                     ]
    table_col_names = ['', 'Title', 'Variables', 'RI', 'Station', 'Station code', 'Start', 'End',
                       #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                       ]
    table_columns = [{'name': name, 'id': i} for name, i in zip(table_col_names, table_col_ids)]
    # on rendering HTML snipplets in DataTable cells:
    # https://github.com/plotly/dash-table/pull/916
    table_columns[0]['presentation'] = 'markdown'

    if datasets_json is None:
        return table_columns, [], [], []

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    if len(datasets_df) > 0:
        datasets_df['time_period_start'] = datasets_df['time_period_start'].dt.strftime('%Y-%m-%d')
        datasets_df['time_period_end'] = datasets_df['time_period_end'].dt.strftime('%Y-%m-%d')

    datasets_df = datasets_df.join(station_by_shortnameRI['long_name'], on='platform_id_RI')

    # filter on selected timeline bars on the Gantt figure
    if gantt_figure_selectedData and 'points' in gantt_figure_selectedData:
        datasets_indices = []
        for timeline_bar in gantt_figure_selectedData['points']:
            datasets_indices.extend(timeline_bar['customdata'][0])
        datasets_df = datasets_df.iloc[datasets_indices]

    # on rendering HTML snipplets in DataTable cells:
    # https://github.com/plotly/dash-table/pull/916
    datasets_df['eye'] = '<i class="fa fa-eye"></i>'

    table_data = datasets_df[['id'] + table_col_ids].to_dict(orient='records')

    # see here for explanation how dash.callback_context works
    # https://community.plotly.com/t/select-all-rows-in-dash-datatable/41466/2
    # TODO: this part needs to be checked and polished;
    # TODO: e.g. is the manual synchronization between selected_rows and selected_row_ids needed?
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if trigger == DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID:
        if datasets_table_checklist_all_none_switch:
            selected_rows = list(range(len(table_data)))
        else:
            selected_rows = []
        selected_row_ids = datasets_df['id'].iloc[selected_rows].to_list()
    else:
        if previously_selected_row_ids is None:
            previously_selected_row_ids = []
        selected_row_ids = sorted(set(previously_selected_row_ids) & set(datasets_df['id'].to_list()))
        idx = pd.DataFrame({'idx': datasets_df['id'], 'n': range(len(datasets_df['id']))}).set_index('idx')
        idx = idx.loc[selected_row_ids]
        selected_row_ids = idx.index.to_list()
        selected_rows = idx['n'].to_list()
    return table_columns, table_data, selected_rows, selected_row_ids


# _active_cell = None


@app.callback(
    Output(QUICKLOOK_POPUP_ID, 'children'),
    #Output(DATASET_MD_STORE_ID, 'data'),
    Input(DATASETS_TABLE_ID, 'active_cell'),
    State(DATASETS_STORE_ID, 'data'),
    prevent_initial_call=True,
)
def popup_graphs(active_cell, datasets_json):
    # global _active_cell

    # _active_cell = active_cell
    # print(f'active_cell={active_cell}')

    if datasets_json is None or active_cell is None:
        return None

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    ds_md = datasets_df.loc[active_cell['row_id']]

    if 'selector' in ds_md and isinstance(ds_md['selector'], str) and len(ds_md['selector']) > 0:
        selector = ds_md['selector']
    else:
        selector = None

    print('selector=', selector)
    try:
        ri = ds_md['RI']
        url = ds_md['url']
        selector = ds_md['selector'] if 'selector' in ds_md else None
        req = data_processing.ReadDataRequest(ri, url, ds_md, selector=selector)
        da_by_var = req.compute()
        # da_by_var = data_access.read_dataset(ds_md['RI'], ds_md['url'], ds_md)
        # for v, da in da_by_var.items():
        #     da.to_netcdf(CACHE_DIR / 'tmp' / f'{v}.nc')
        ds_exc = None
    except Exception as e:
        da_by_var = None
        ds_exc = e

    if da_by_var is not None:
        ds_vars = [v for v in da_by_var if da_by_var[v].squeeze().ndim == 1]
        if len(ds_vars) > 0:
            ds_plot = dcc.Graph(
                id='quick-plot',
                figure=_plot_vars(da_by_var, ds_vars[0], ds_vars[1] if len(ds_vars) > 1 else None),
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': False,
                }
            )
        else:
            ds_plot = None
    else:
        ds_plot = repr(ds_exc)

    popup = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(ds_md['title'])),
            dbc.ModalBody(children=[
                ds_plot,
                # html.Button('Download CSV', id='btn_csv'),
                # dcc.Download(id='download_csv'),
            ]),
        ],
        id="modal-xl",
        size="xl",
        is_open=True,
    )

    return popup #, ds_md.to_json(orient='index', date_format='iso')


# @app.callback(
#     Output('download_csv', 'data'),
#     Input('btn_csv', 'n_clicks'),
#     State(DATASET_MD_STORE_ID, 'data'),
#     prevent_initial_call=True,
# )
def download_csv(n_clicks, ds_md_json):
    try:
        s = pd.Series(json.loads(ds_md_json))
        ds = data_access.read_dataset(s['RI'], s['url'], s)
        df = ds.reset_coords(drop=True).to_dataframe()
        download_filename = werkzeug.utils.secure_filename(s['title'] + '.csv')
        return dcc.send_data_frame(df.to_csv, download_filename)
    except Exception as e:
        logger().exception(f'Failed to download the dataset {ds_md_json}', exc_info=e)


@app.callback(
    Output(SELECT_DATASETS_REQUEST_ID, 'data'),
    # Output(AVAIL_DATA_GRAPH_ID, 'figure'),
    # Output(VAR_HIST_GRAPH_1_ID, 'figure'),
    # Output(VAR_HIST_GRAPH_2_ID, 'figure'),
    # Output(VAR_HIST_GRAPH_3_ID, 'figure'),
    Input(SELECT_DATASETS_BUTTON_ID, 'n_clicks'),
    State(DATASETS_STORE_ID, 'data'),
    State(DATASETS_TABLE_ID, 'selected_row_ids'),
    prevent_initial_call=True,
)
def select_datasets(n_clicks, datasets_json, selected_row_ids):
    if datasets_json is None or selected_row_ids is None:
        return None

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    ds_md = datasets_df.loc[selected_row_ids]
    read_dataset_requests = []
    for idx, ds_metadata in ds_md.iterrows():
        ri = ds_metadata['RI']
        url = ds_metadata['url']
        selector = ds_metadata['selector'] if 'selector' in ds_metadata else None
        req = data_processing.ReadDataRequest(ri, url, ds_metadata, selector=selector)
        # req.compute()  ###
        read_dataset_requests.append(req)

    req = data_processing.IntegrateDatasetsRequest(read_dataset_requests)
    # TODO: do it asynchronously? will it work with dash/flask? look at options of @app.callback decorator (background=True, ???)
    req.compute()
    # ds = req.compute()
    # avail_data_by_var_plot = get_avail_data_by_var(ds)
    # df = ds.to_dataframe()
    # hists = []
    # for i, v in enumerate(list(df)[:3]):
    #     *v_label, ri = v.split('_')
    #     v_label = '_'.join(v_label)
    #     hist_fig = px.histogram(
    #         df, x=v, nbins=100,
    #         title=f'{v_label} ({ri})',
    #         labels={v: f'{v_label} ({ri})'},
    #     )
    #     hist_fig.update_layout(
    #         clickmode='event',
    #         selectdirection='h',
    #     )
    #     hists.append(hist_fig)
    # if len(hists) < 3:
    #     hists.extend((None, ) * (3 - len(hists)))
    # return req.to_dict(), avail_data_by_var_plot, hists[0], hists[1], hists[2]
    return req.to_dict()


from utils import crossfiltering  # noq (install callbacks)


# @app.callback(
#     Output(AVAIL_DATA_GRAPH_ID, 'figure'),
#     Output(VAR_HIST_GRAPHS_CONTAINER_ID, 'children'),
#     Input(SELECT_DATASETS_REQUEST_ID, 'data'),
#     prevent_initial_call=True,
# )
# def get_data_histograms(select_datasets_request):
#     if select_datasets_request is not None:
#         req = data_processing.MergeDatasetsRequest.from_dict(select_datasets_request)
#         ds = req.compute()
#         avail_data_by_var_plot = get_avail_data_by_var_gantt(ds)
#         df = ds.to_dataframe()
#         hists = []
#         for i, v in enumerate(list(df)):
#             *v_label, ri = v.split('_')
#             v_label = '_'.join(v_label)
#             hist_fig = px.histogram(
#                 df, x=v, nbins=100,
#                 title=f'{v_label} ({ri})',
#                 labels={v: f'{v_label} ({ri})'},
#             )
#             hist_fig.update_layout(
#                 clickmode='event',
#                 selectdirection='h',
#             )
#             hists.append(dcc.Graph(
#                 id={'type': VAR_HIST_GRAPHS_CONTAINER_ID, 'index': i},
#                 figure=hist_fig,
#             ))
#         return avail_data_by_var_plot, hists
#     else:
#         return None, None


# @app.callback(
#     Output(SELECT_DATASETS_REQUEST_ID, 'data'),
#     Output(TIME_HISTOGRAM_GRAPH_ID, 'figure'),
#     Output(SCATTER_GRAPH_ID, 'figure'),
#     Input(SELECT_DATASETS_BUTTON_ID, 'n_clicks'),
#     State(DATASETS_STORE_ID, 'data'),
#     State(DATASETS_TABLE_ID, 'selected_row_ids'),
#     prevent_initial_call=True,
# )
# def select_datasets(n_clicks, datasets_json, selected_row_ids):
#     if datasets_json is None or selected_row_ids is None:
#         return None
#
#     datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
#     ds_md = datasets_df.loc[selected_row_ids]
#     read_dataset_requests = []
#     for idx, ds_metadata in ds_md.iterrows():
#         ri = ds_metadata['RI']
#         url = ds_metadata['url']
#         selector = ds_metadata['selector'] if 'selector' in ds_metadata else None
#         req = data_processing.ReadDataRequest(ri, url, ds_metadata, selector=selector)
#         # req.compute()  ###
#         read_dataset_requests.append(req)
#
#     req = data_processing.MergeDatasetsRequest(read_dataset_requests)
#     # req.compute()  # TODO: do it lazy ???
#     ds = req.compute()
#     line_plot = _get_line_plot(ds)
#     scatter_plot = _get_scatter_plot(ds)
#     return req.to_dict(), line_plot, scatter_plot


# End of callback definitions


# Launch the Dash application.
# app_conf['debug'] = False
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
