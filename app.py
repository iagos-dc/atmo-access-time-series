"""
ATMO-ACCESS time series service
"""

import numpy as np
import pandas as pd
import plotly.express as px

# Dash imports; for documentation (including tutorial), see: https://dash.plotly.com/
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State

# Provides a version of Dash application which can be run in Jupyter notebook/lab
# See: https://github.com/plotly/jupyter-dash
from jupyter_dash import JupyterDash

# Local imports
import data_access


# Configuration of the app
# See: https://dash.plotly.com/devtools#configuring-with-run_server
# for the usual Dash app, and:
# https://github.com/plotly/jupyter-dash/blob/master/notebooks/getting_started.ipynb
# for a JupyterDash app version.
RUNNING_IN_BINDER = False
app_conf = {'mode': 'external', 'debug': True}
if RUNNING_IN_BINDER:
    JupyterDash.infer_jupyter_proxy_config()
else:
    app_conf.update({'host': 'localhost', 'port': 9235})


# Below there are id's of Dash JS components.
# The components themselves are declared in the dashboard layout (see the function get_dashboard_layout).
# Essential properties of each component are explained in the comments below.
STATIONS_MAP_ID = 'stations-map'
    # 'selectedData' contains a dictionary
    # {
    #   'point' ->
    #       list of dicionaries {'pointIndex' -> index of a station in the global dataframe stations, 'lon' -> float, 'lat' -> float, ...},
    #   'range' (present only if a box was selected on the map) ->
    #       {'mapbox' -> [[lon_min, lat_max], [lon_max, lat_min]]}
    # }
VARIABLES_CHECKLIST_ID = 'variables-checklist'
SELECTED_STATIONS_DROPDOWN_ID = 'selected-stations-dropdown'
    # 'options' contains a list of dictionaries {'label' -> station label, 'value' -> index of the station in the global dataframe stations (see below)}
    # 'value' contains a list of indices of stations selected using the dropdown
SEARCH_DATASETS_BUTTON_ID = 'search-datasets-button'
    # 'n_click' contains a number of click at the button
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
DATASETS_TABLE_ID = 'datasets-table'
    # 'columns' contains list of dictionaries {'name' -> column name, 'id' -> column id}
    # 'data' contains a list of records as provided by the method pd.DataFrame.to_dict(orient='records')


# Initialization of global objects
app = JupyterDash(__name__)
stations = data_access.get_stations()
variables = data_access.get_vars()


# Begin of definition of routines which constructs components of the dashboard

def get_variables_checklist():
    """
    Provide variables checklist Dash component
    See: https://dash.plotly.com/dash-core-components/checklist
    :return: dash.dcc.Checklist
    """
    variables_options = variables \
        .rename(columns={'ECV_name': 'label', 'variable_name': 'value'})[['label', 'value']] \
        .to_dict(orient='index').values()
    variables_checklist = dcc.Checklist(
        id=VARIABLES_CHECKLIST_ID,
        options=list(variables_options),
        value=list(variables['variable_name']),
        labelStyle={'display': 'flex'},  # display in column rather than in a row; not sure if it is the right way to do
    )
    return variables_checklist


def get_stations_map():
    """
    Provide a Dash component containing a map with stations
    See: https://dash.plotly.com/dash-core-components/graph
    :return: dash.dcc.Graph object
    """
    fig = px.scatter_mapbox(
        stations,
        lat="latitude", lon="longitude", color='RI',
        hover_name="long_name", hover_data=["ground_elevation"],
        color_discrete_sequence=["red"], #, "blue"],  # can define colors to be used in case of many RI's
        zoom=3, height=600
    )
    fig.update_layout(mapbox_style="open-street-map",
                      margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
                      clickmode='event+select')

    stations_map = dcc.Graph(
        id=STATIONS_MAP_ID,
        figure=fig,
    )
    return stations_map


def get_bbox_selection_div():
    """
    Provide a composed Dash component with input/ouput text fields which allow to provide coordinates of a bounding box
    See: https://dash.plotly.com/dash-core-components/input
    :return: dash.html.Div object
    """
    bbox_selection_div = html.Div(id='bbox-selection-div', children=[
        html.Div(className='row', children=[
            html.Div(className='offset-by-six columns', children=[
                dcc.Input(className='three columns', id=LAT_MAX_ID, placeholder='lat max', type='number', min=-90,
                          max=90),  # , step=0.01),
            ]),
        ]),
        html.Div(className='row', children=[
            html.P(className='three columns', children='Bounding box:', style={'font-weight': 'bold'}),
            dcc.Input(className='three columns', id=LON_MIN_ID, placeholder='lon min', type='number', min=-180,
                      max=180),  # , step=0.01),
            dcc.Input(className='three columns, offset-by-three columns', id=LON_MAX_ID, placeholder='lon max',
                      type='number', min=-180, max=180),  # , step=0.01),
        ]),
        html.Div(className='row', children=[
            html.Div(className='offset-by-six columns', children=[
                dcc.Input(className='three columns', id=LAT_MIN_ID, placeholder='lat min', type='number', min=-90,
                          max=90),  # , step=0.01),
            ]),
        ]),
    ])
    return bbox_selection_div


def get_dashboard_layout():
    # these are special Dash components used for transferring data from one callback to other callback(s)
    # without displaying the data
    stores = [
        dcc.Store(id=DATASETS_STORE_ID),
    ]

    layout = html.Div(id='app-container-div', children=stores + [
        html.Div(id='left-panel-div', className='four columns', children=[

            # logo and application title
            html.Div(className='row', children=[
                html.Div(style={'float': 'left'}, children=[
                    html.A(
                        html.Img(
                            src=app.get_asset_url('atmo_access_logo.png'),
                            style={'float': 'right', 'height': '40px', 'margin-top': '10px'}
                        ),
                        href="https://www.atmo-access.eu/",
                    ),
                ]),
                html.Div(style={'float': 'right'}, children=[
                    html.H3('Time-series analysis'),
                ]),
            ]),

            html.Div(id='variables-selection-div', children=[
                html.P('Select variable(s)', style={'font-weight': 'bold'}),
                get_variables_checklist(),
            ]),

            get_bbox_selection_div(),

            html.Hr(),

            html.Div(id='selected-stations-div', children=[
                html.P('Selected stations (you can refine your selection)', style={'font-weight': 'bold'}),
                dcc.Dropdown(id=SELECTED_STATIONS_DROPDOWN_ID, multi=True, clearable=False),
            ]),

            html.Hr(),

            html.Button(id=SEARCH_DATASETS_BUTTON_ID, n_clicks=0, children='Search datasets'),
        ]),

        html.Div(id='rest-of-dashboard-div', className='eight columns', children=[
            get_stations_map(),

            html.Hr(),

            dcc.RadioItems(
                id=GANTT_VIEW_RADIO_ID,
                options=[
                    {'label': 'compact view', 'value': 'compact'},
                    {'label': 'detailed view', 'value': 'detailed'},
                ],
                value='compact',
                labelStyle={'display': 'flex'},
            ),

            dcc.Graph(
                id=GANTT_GRAPH_ID,
            ),

            dash_table.DataTable(
                id=DATASETS_TABLE_ID,
            ),
        ]),
    ])
    return layout

# End of definition of routines which constructs components of the dashboard


# Assign a dashboard layout to app Dash object
app.layout = get_dashboard_layout()


# Begin of callback definitions and their helper routines.
# See: https://dash.plotly.com/basic-callbacks
# for a basic tutorial and
# https://dash.plotly.com/  -->  Dash Callback in left menu
# for more detailed documentation

@app.callback(
    Output(DATASETS_STORE_ID, 'data'),
    Input(SEARCH_DATASETS_BUTTON_ID, 'n_clicks'),
    State(VARIABLES_CHECKLIST_ID, 'value'),
    State(LON_MIN_ID, 'value'),
    State(LON_MAX_ID, 'value'),
    State(LAT_MIN_ID, 'value'),
    State(LAT_MAX_ID, 'value'),
    State(SELECTED_STATIONS_DROPDOWN_ID, 'value')
)
def search_datasets(n_clicks, selected_variables, lon_min, lon_max, lat_min, lat_max, selected_stations_idx):
    if selected_stations_idx is None:
        selected_stations_idx = []
    datasets_df = data_access.get_datasets(selected_variables, lon_min, lon_max, lat_min, lat_max)
    selected_stations_short_name = stations['short_name'].iloc[selected_stations_idx]
    datasets_df_filtered = datasets_df[datasets_df['platform_id'].isin(selected_stations_short_name)]
    return datasets_df_filtered.to_json(orient='split', date_format='iso')


@app.callback(
    Output(DATASETS_TABLE_ID, 'columns'),
    Output(DATASETS_TABLE_ID, 'data'),
    Input(DATASETS_STORE_ID, 'data'),
)
def datasets_as_table(datasets_json):
    if not datasets_json:
        return None, None
    datasets_df = pd.read_json(datasets_json, orient='split')
    table_columns = [{"name": i, "id": i} for i in datasets_df.columns]
    table_data = datasets_df.to_dict(orient='records')
    return table_columns, table_data


def _get_selected_points(selected_stations):
    if selected_stations is not None:
        points = selected_stations['points']
    else:
        points = []
    return pd.DataFrame.from_records(points, index='pointIndex', columns=['pointIndex', 'lon', 'lat'])


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


def _get_selected_stations_option_list(selected_stations_df):
    idx = selected_stations_df.index
    df = stations.iloc[idx]
    labels = df['short_name'] + ' (' + df['long_name'] + ', ' + df['country'] + ')'
    options = labels.rename('label').reset_index()
    return [html.Option(children=option, value=str(i), selected=True) for i, option in zip(options['index'], options['label'])]


def _get_selected_stations_dropdown(selected_stations_df):
    idx = selected_stations_df.index
    df = stations.iloc[idx]
    labels = df['short_name'] + ' (' + df['long_name'] + ', ' + df['country'] + ')'
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


def _contiguous_periods(start, end):
    s, e, idx = [], [], []
    df = pd.DataFrame({'s': start, 'e': end}).sort_values(by='s', ignore_index=False)
    df['e'] = df['e'].cummax()
    if len(df) > 0:
        delims, = np.nonzero(df['e'].values[:-1] < df['s'].values[1:])
        delims = np.concatenate(([0], delims + 1, [len(df)]))
        for i, j in zip(delims[:-1], delims[1:]):
            s.append(df['s'].iloc[i])
            e.append(df['e'].iloc[j - 1])
            idx.append(df.index[i:j])
    return pd.DataFrame({'time_period_start': s, 'time_period_end': e, 'idx': idx})


def _get_timeline_by_station(datasets_df):
    df = datasets_df\
        .groupby(['platform_id'])\
        .apply(lambda x: _contiguous_periods(x.time_period_start, x.time_period_end))\
        .reset_index()
    no_platforms = len(df['platform_id'].unique())
    height = 100 + max(100, 50 + 30 * no_platforms)
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y='platform_id', color='platform_id',
        height=height
    )
    gantt.update_layout(clickmode='event+select')
    return gantt


def _get_timeline_by_station_and_vars(datasets_df):
    df = datasets_df\
        .groupby(['platform_id', 'var_codes_filtered'])\
        .apply(lambda x: _contiguous_periods(x.time_period_start, x.time_period_end))\
        .reset_index()
    facet_col_wrap = 3
    no_platforms = len(df['platform_id'].unique())
    no_var_codes_filtered = len(df['var_codes_filtered'].unique())
    no_facet_rows = (no_var_codes_filtered + facet_col_wrap - 1) // facet_col_wrap
    height = 100 + max(100, 50 + 25 * no_platforms) * no_facet_rows
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y='platform_id', color='var_codes_filtered',
        height=height, facet_col='var_codes_filtered', facet_col_wrap=facet_col_wrap,
    )
    gantt.update_layout(clickmode='event+select')
    return gantt


@app.callback(
    Output(GANTT_GRAPH_ID, 'figure'),
    Input(GANTT_VIEW_RADIO_ID, 'value'),
    Input(DATASETS_STORE_ID, 'data'),
)
def get_gantt_figure(gantt_view_type, datasets_json):
    datasets_df = pd.read_json(datasets_json, orient='split')
    if len(datasets_df) == 0:
        return {}   # empty figure; TODO: is it the right way to do?
    if gantt_view_type == 'compact':
        return _get_timeline_by_station(datasets_df)
    else:
        return _get_timeline_by_station_and_vars(datasets_df)

# End of callback definitions


# Launch the Dash application.
app.run_server(**app_conf)
