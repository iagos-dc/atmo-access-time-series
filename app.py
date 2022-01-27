"""
ATMO-ACCESS time series service
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Dash imports; for documentation (including tutorial), see: https://dash.plotly.com/
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

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
RUNNING_IN_BINDER = False   # for running in Binder change it to True
app_conf = {'mode': 'external', 'debug': True}  # for running inside a Jupyter notebook change 'mode' to 'inline'
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
QUICKLOOK_POPUP_ID = 'quicklook-popup'
    # 'children' contains a layout of the popup

# Color codes
ACTRIS_COLOR_HEX = '#00adb7'
IAGOS_COLOR_HEX = '#456096'
ICOS_COLOR_HEX = '#ec165c'


def _get_station_by_shortnameRI(stations):
    df = stations.set_index('short_name_RI')[['long_name', 'RI']]
    df['station_fullname'] = df['long_name'] + ' (' + df['RI'] + ')'
    return df


# Initialization of global objects
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
stations = data_access.get_stations()
station_by_shortnameRI = _get_station_by_shortnameRI(stations)
variables = data_access.get_vars()


# Begin of definition of routines which constructs components of the dashboard

def get_variables_checklist():
    """
    Provide variables checklist Dash component
    See: https://dash.plotly.com/dash-core-components/checklist
    :return: dash.dcc.Checklist
    """
    std_vars = variables[['std_ECV_name', 'code']].drop_duplicates()
    # TODO: temporary
    try:
        std_vars = std_vars[std_vars['std_ECV_name'] != 'Aerosol Optical Properties']
    except ValueError:
        pass
    std_vars['label'] = std_vars['code'] + ' - ' + std_vars['std_ECV_name']
    std_vars = std_vars.rename(columns={'std_ECV_name': 'value'}).drop(columns='code')
    variables_options = std_vars.to_dict(orient='records')
    variables_checklist = dbc.Checklist(
        id=VARIABLES_CHECKLIST_ID,
        options=variables_options,
        value=std_vars['value'].tolist(),
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
        hover_name="long_name", hover_data={'ground_elevation': True, 'marker_size': False},
        custom_data=['idx'],
        size=stations['marker_size'],
        size_max=7,
        category_orders={'RI': ['ACTRIS', 'IAGOS', 'ICOS']},
        color_discrete_sequence=[ACTRIS_COLOR_HEX, IAGOS_COLOR_HEX, ICOS_COLOR_HEX],
        zoom=3, height=600,
        title='Stations map',
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={'r': 0, 't': 0, 'l': 0, 'b': 0},
        clickmode='event+select',
        hoverdistance=1, hovermode='closest',  # hoverlabel=None,
    )

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

            dbc.RadioItems(
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
        ]),

        html.Div(id='table-div', className='twelve columns', children=[
            dash_table.DataTable(
                id=DATASETS_TABLE_ID,
                # see: https://dash.plotly.com/datatable/interactivity
                row_selectable="multi",
                selected_rows=[],
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
            ),
        ]),

        html.Div(id=QUICKLOOK_POPUP_ID),
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


def _contiguous_periods(start, end, var_codes=None, dt=pd.Timedelta('1D')):
    """
    Merge together periods which overlap, are adjacent or nearly adjacent (up to dt). The merged periods are returned
    with:
    - start and end time ('time_period_start', 'time_period_end'),
    - list of indices of datasets which enters into a given period ('indices'),
    - number of the datasets (the length of the above list) ('datasets'),
    - codes of variables available within a given period, if the parameter var_codes is provided.
    :param start: pandas.Series of Timestamps with periods' start
    :param end: pandas.Series of Timestamps with periods' end
    :param var_codes: pandas.Series of strings or None, optional; if given, must contain variable codes separated by comma
    :param dt: pandas.Timedelta
    :return: pandas.DataFrame with columns 'time_period_start', 'time_period_end', 'indices', 'datasets' and 'var_codes'
    """
    s, e, idx = [], [], []
    df_dict = {'s': start, 'e': end}
    if var_codes is not None:
        dat = []
        df_dict['var_codes'] = var_codes
    df = pd.DataFrame(df_dict).sort_values(by='s', ignore_index=False)
    df['e'] = df['e'].cummax()
    if len(df) > 0:
        delims, = np.nonzero((df['e'] + dt).values[:-1] < df['s'].values[1:])
        delims = np.concatenate(([0], delims + 1, [len(df)]))
        for i, j in zip(delims[:-1], delims[1:]):
            s.append(df['s'].iloc[i])
            e.append(df['e'].iloc[j - 1])
            idx.append(df.index[i:j])
            if var_codes is not None:
                # concatenate all var_codes; [:-1] is to suppress the last comma
                all_var_codes = (df['var_codes'].iloc[i:j] + ', ').sum()[:-2]
                # remove duplicates from all_var_codes...
                all_var_codes = np.sort(np.unique(all_var_codes.split(', ')))
                # ...and form a single string with codes separated by comma
                all_var_codes = ', '.join(all_var_codes)
                dat.append(all_var_codes)
    res_dict = {'time_period_start': s, 'time_period_end': e, 'indices': idx, 'datasets': [len(i) for i in idx]}
    if var_codes is not None:
        res_dict['var_codes'] = dat
    return pd.DataFrame(res_dict)


def _get_timeline_by_station(datasets_df):
    df = datasets_df\
        .groupby(['platform_id_RI', 'station_fullname', 'RI'])\
        .apply(lambda x: _contiguous_periods(x['time_period_start'], x['time_period_end'], x['var_codes_filtered']))\
        .reset_index()
    df = df.sort_values('platform_id_RI')
    no_platforms = len(df['platform_id_RI'].unique())
    height = 100 + max(100, 50 + 30 * no_platforms)
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y='platform_id_RI', color='RI',
        hover_name='var_codes',
        hover_data={'station_fullname': True, 'platform_id_RI': True, 'datasets': True, 'RI': False},
        custom_data=['indices'],
        category_orders={'RI': ['ACTRIS', 'IAGOS', 'ICOS']},
        color_discrete_sequence=[ACTRIS_COLOR_HEX, IAGOS_COLOR_HEX, ICOS_COLOR_HEX],
        height=height
    )
    gantt.update_layout(clickmode='event+select')
    return gantt


def _get_timeline_by_station_and_vars(datasets_df):
    df = datasets_df\
        .groupby(['platform_id_RI', 'station_fullname', 'var_codes_filtered'])\
        .apply(lambda x: _contiguous_periods(x['time_period_start'], x['time_period_end']))\
        .reset_index()
    df = df.sort_values('platform_id_RI')
    facet_col_wrap = 3
    no_platforms = len(df['platform_id_RI'].unique())
    no_var_codes_filtered = len(df['var_codes_filtered'].unique())
    no_facet_rows = (no_var_codes_filtered + facet_col_wrap - 1) // facet_col_wrap
    height = 100 + max(100, 50 + 25 * no_platforms) * no_facet_rows
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y='platform_id_RI', color='var_codes_filtered',
        hover_name='station_fullname',
        hover_data={'station_fullname': True, 'platform_id_RI': True, 'var_codes_filtered': True, 'datasets': True},
        custom_data=['indices'],
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
    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    datasets_df = datasets_df.join(station_by_shortnameRI['station_fullname'], on='platform_id_RI')  # column 'station_fullname' joined to datasets_df
    if len(datasets_df) == 0:
        return {}   # empty figure; TODO: is it the right way to do?
    if gantt_view_type == 'compact':
        return _get_timeline_by_station(datasets_df)
    else:
        return _get_timeline_by_station_and_vars(datasets_df)


@app.callback(
    Output(DATASETS_TABLE_ID, 'columns'),
    Output(DATASETS_TABLE_ID, 'data'),
    Output(DATASETS_TABLE_ID, 'selected_rows'),
    Input(DATASETS_STORE_ID, 'data'),
    Input(GANTT_GRAPH_ID, 'selectedData'),
)
def datasets_as_table(datasets_json, gantt_figure_selectedData):
    if not datasets_json:
        return None, None
    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    datasets_df = datasets_df.join(station_by_shortnameRI['long_name'], on='platform_id_RI')

    # print(gantt_figure_selectedData)
    # filter on selected timeline bars on the Gantt figure
    if gantt_figure_selectedData and 'points' in gantt_figure_selectedData:
        datasets_indices = []
        for timeline_bar in gantt_figure_selectedData['points']:
            datasets_indices.extend(timeline_bar['customdata'][0])
        datasets_df = datasets_df.iloc[datasets_indices]

    table_col_ids = ['title', 'var_codes_filtered', 'RI', 'long_name', 'platform_id', 'time_period_start', 'time_period_end',
                     #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                     ]
    table_col_names = ['Title', 'Variables', 'RI', 'Station', 'Station code', 'Start', 'End',
                       #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                       ]
    table_columns = [{'name': name, 'id': i} for name, i in zip(table_col_names, table_col_ids)]
    table_data = datasets_df[['id'] + table_col_ids].to_dict(orient='records')
    selected_rows = []
    return table_columns, table_data, selected_rows


_tmp_dataset_df = None
_tmp_ds = None
_selected_row_ids = None


def _plot_vars(ds, v1, v2=None):
    vars_long = data_access.get_vars_long()
    vs = [v1, v2] if v2 is not None else [v1]
    v_names = []
    for v in vs:
        try:
            v_name = vars_long.loc[vars_long['variable_name'] == v]['std_ECV_name'].iloc[0] + f' ({v})'
        except:
            v_name = v
        v_names.append(v_name)
    fig = go.Figure()
    for i, v in enumerate(vs):
        da = ds[v]
        fig.add_trace(go.Scatter(
            x=da['time'].values,
            y=da.values,
            name=v,
            yaxis=f'y{i + 1}'
        ))

    fig.update_layout(
        xaxis=dict(
            domain=[0.0, 0.95]
        ),
        yaxis1=dict(
            title=v_names[0],
            titlefont=dict(
                color="#1f77b4"
            ),
            tickfont=dict(
                color="#1f77b4"
            ),
            anchor='x',
            side='left',
        ),
    )
    if v2 is not None:
        fig.update_layout(
            yaxis2=dict(
                title=v_names[1],
                titlefont=dict(
                    color="#ff7f0e"
                ),
                tickfont=dict(
                    color="#ff7f0e"
                ),
                anchor="x",
                overlaying="y1",
                side="right",
                # position=0.15
            ),
        )

    return fig


@app.callback(
    Output(QUICKLOOK_POPUP_ID, 'children'),
    Input(DATASETS_TABLE_ID, 'selected_row_ids'),
    State(DATASETS_STORE_ID, 'data'),
)
def popup_graphs(selected_row_ids, datasets_json):
    global _tmp_dataset_df, _tmp_ds, _selected_row_ids

    _selected_row_ids = selected_row_ids

    if datasets_json is None or selected_row_ids is None:
        return None
    if len(selected_row_ids) == 0:
        return []

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    datasets_df = datasets_df.loc[selected_row_ids]
    _tmp_dataset_df = datasets_df

    s = datasets_df.iloc[-1]
    ds = data_access.read_dataset(s['RI'], s['url'], s)
    _tmp_ds = ds

    ds_vars = [v for v in ds if ds[v].squeeze().ndim == 1]

    return dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(s['title'])),
            dbc.ModalBody(
                dcc.Graph(
                    id='quick-plot',
                    figure=_plot_vars(ds, ds_vars[0], ds_vars[1] if len(ds_vars) > 1 else None)
                ) if len(ds_vars) > 0 else None
            ),
        ],
        id="modal-xl",
        size="xl",
        is_open=True,
    )
    #print(df['RI'][0] + ': ' + str(df['url'][0]))
    #return df['RI'] + ': ' + str(df['url'])

# End of callback definitions


# Launch the Dash application.
app.run_server(**app_conf)
