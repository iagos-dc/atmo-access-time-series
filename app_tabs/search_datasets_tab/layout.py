import datetime

import dash_bootstrap_components as dbc
from dash import dcc, html

import data_access
import utils.stations_map
from app_tabs.common.layout import SEARCH_DATASETS_TAB_VALUE
from utils.dash_persistence import get_dash_persistence_kwargs

VARIABLES_CHECKLIST_ID = 'variables-checklist'

STATIONS_MAP_ID = 'stations-map'
DEFAULT_MAP_ZOOM = 2
MAP_ZOOM_STORE_ID = 'previous-map-zoom-store'

# 'selectedData' contains a dictionary
# {
#   'point' ->
#       list of dicionaries {'pointIndex' -> index of a station in the global dataframe stations, 'lon' -> float, 'lat' -> float, ...},
#   'range' (present only if a box was selected on the map) ->
#       {'mapbox' -> [[lon_min, lat_max], [lon_max, lat_min]]}
# }

LAT_MAX_ID = 'lat-max'
LAT_MIN_ID = 'lat-min'
LON_MAX_ID = 'lon-max'
LON_MIN_ID = 'lon-min'
# 'value' contains a number (or None)

VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'variables-checklist-all-none-switch'

SELECTED_STATIONS_DROPDOWN_ID = 'selected-stations-dropdown'
# 'options' contains a list of dictionaries {'label' -> station label, 'value' -> index of the station in the global dataframe stations (see below)}
# 'value' contains a list of indices of stations selected using the dropdown

SEARCH_DATASETS_BUTTON_ID = 'search-datasets-button'
# 'n_click' contains a number of clicks at the button

SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID = 'search-datasets-reset-stations-button'

DATE_RANGE_PICKER_ID = 'search-datasets-date-picker-range'

MAP_BACKGROUND_RADIO_ID = 'map-background-radio'

MAPBOX_STYLES = {
    'open-street-map': 'open street map',
    'carto-positron': 'carto positron',
}
DEFAULT_MAPBOX_STYLE = 'carto-positron'


def _get_std_variables(variables):
    std_vars = variables[['std_ECV_name', 'code']].drop_duplicates()
    # TODO: temporary
    try:
        std_vars = std_vars[std_vars['std_ECV_name'] != 'Aerosol Optical Properties']
    except ValueError:
        pass
    std_vars['label'] = std_vars['code'] + ' - ' + std_vars['std_ECV_name']
    return std_vars.rename(columns={'std_ECV_name': 'value'}).drop(columns='code')


variables = data_access.get_vars()
std_variables = _get_std_variables(variables)


# TODO: variable list should be loaded periodically via a callback
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
        persistence=True,
        persistence_type='session',
    )
    return variables_checklist


def get_stations_dash_component():
    fig = utils.stations_map.get_stations_map(DEFAULT_MAP_ZOOM)
    fig.update_layout(
        mapbox=dict(
            style=DEFAULT_MAPBOX_STYLE,
            zoom=DEFAULT_MAP_ZOOM,
            center={'lon': 10, 'lat': 55},
        ),
        title='Stations map',
        margin={'autoexpand': True, 'r': 0, 't': 40, 'l': 0, 'b': 0},
        # width=1100, height=700,
        autosize=True,
        clickmode='event+select',
        dragmode='select',
        hoverdistance=1, hovermode='closest',  # hoverlabel=None,
        selectionrevision=False,  # this is crucial !!!
    )

    stations_map = dcc.Graph(
        id=STATIONS_MAP_ID,
        figure=fig,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'scrollZoom': True,
        }
    )

    return dbc.Card(
        dbc.CardBody([
            dcc.Store(id=MAP_ZOOM_STORE_ID, data=DEFAULT_MAP_ZOOM),
            stations_map
        ])
    )


def get_bbox_selection_div():
    """
    Provide a composed Dash component with input/ouput text fields which allow to provide coordinates of a bounding box
    See: https://dash.plotly.com/dash-core-components/input
    :return: dash.html.Div object
    """
    bbox_selection_div = html.Div(id='bbox-selection-div', style={'margin-top': '15px'}, children=[
        html.Div(className='row', children=[
            html.Div(
                className='three columns, offset-by-six columns',
                children=dcc.Input(
                    id=LAT_MAX_ID,
                    style={'width': '120%'},
                    placeholder='lat max',
                    type='number',
                    debounce=True,
                    min=-90, max=90,
                    **get_dash_persistence_kwargs(persistence_id=True)
                ),  # , step=0.01),
            ),
        ]),
        html.Div(className='row', children=[
            html.Div(className='three columns',
                     children=html.P(children='Bounding box:', style={'width': '100%', 'font-weight': 'bold'})),
            html.Div(
                className='three columns',
                children=dcc.Input(
                    style={'width': '120%'},
                    id=LON_MIN_ID,
                    placeholder='lon min',
                    type='number',
                    debounce=True,
                    min=-180, max=180,
                    **get_dash_persistence_kwargs(persistence_id=True)
                ),  # , step=0.01),
            ),
            html.Div(
                className='offset-by-three columns, three columns',
                children=dcc.Input(
                    style={'width': '120%'},
                    id=LON_MAX_ID,
                    placeholder='lon max',
                    type='number',
                    debounce=True,
                    min=-180, max=180,
                    **get_dash_persistence_kwargs(persistence_id=True)
                ),  # , step=0.01),
            ),
        ]),
        html.Div(
            className='row',
            children=html.Div(
                className='offset-by-six columns, three columns',
                children=dcc.Input(
                    style={'width': '120%'},
                    id=LAT_MIN_ID,
                    placeholder='lat min',
                    type='number',
                    debounce=True,
                    min=-90, max=90,
                    **get_dash_persistence_kwargs(persistence_id=True)
                ),  # , step=0.01),
            ),
        ),
    ])
    return bbox_selection_div


def get_search_datasets_tab():
    return dbc.Tab(
        label='2. Search datasets',
        id=SEARCH_DATASETS_TAB_VALUE,
        tab_id=SEARCH_DATASETS_TAB_VALUE,
        #value=SEARCH_DATASETS_TAB_VALUE,
        children=html.Div(
            style={'margin': '20px'},
            children=[
                html.Div(id='search-datasets-left-panel-div', className='three columns', children=dbc.Card(dbc.CardBody([
                    html.P('Select variable(s):', style={'font-weight': 'bold'}),
                    dbc.Switch(
                        id=VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID,
                        label='Select all / none',
                        style={'margin-top': '10px'},
                        value=True,
                    ),
                    get_variables_checklist(),
                    #style={'margin-top': '20px'},
                    # html.Div(children=[
                    #     html.P(
                    #         'Date range:',
                    #         style={'display': 'inline', 'font-weight': 'bold', 'margin-right': '20px'}
                    #     ),
                    #     dcc.DatePickerRange(
                    #         id=DATE_RANGE_PICKER_ID,
                    #         min_date_allowed=datetime.date(1900, 1, 1),
                    #         max_date_allowed=datetime.date(2100, 12, 31),
                    #         initial_visible_month=datetime.date.today(),
                    #         display_format='YYYY-MM-DD',
                    #         clearable=True,
                    #         **get_dash_persistence_kwargs(persistence_id=True),
                    #     ),
                    # ]),
                    # get_bbox_selection_div(),
                ]))),
                html.Div(id='search-datasets-right-panel-div', className='nine columns', children=[
                    dbc.Row(
                        dbc.Col(
                            children=[
                                html.Div(
                                    dbc.Button(
                                        id=SEARCH_DATASETS_BUTTON_ID,
                                        n_clicks=0,
                                        color='success',
                                        #color='primary',
                                        type='submit',
                                        style={'font-weight': 'bold', 'font-size': '100%'},
                                        children=html.Div([
                                            'Apply ',
                                            html.I(className='fa fa-arrow-circle-right fa-2x')
                                        ]),
                                        className='me-1',
                                    ),
                                ),
                            ],
                            width='auto',
                        ),
                        justify='end',
                        style={'margin-bottom': '10px'},
                    ),
                    dbc.Row(
                        dbc.Col(
                            get_stations_dash_component(),
                        ),
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Button(
                                    id=SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID,
                                    color='primary',
                                    type='submit',
                                    style={'font-weight': 'bold', 'margin-bottom': '10px'},
                                    children='Clear station selection',
                                ),
                                width='auto',
                            ),
                            dbc.Col(
                                dbc.InputGroup([
                                    dbc.InputGroupText('Map background: ', style={'margin-right': '10px'}),
                                    dbc.RadioItems(
                                        id=MAP_BACKGROUND_RADIO_ID,
                                        options=[
                                            {'label': label, 'value': value}
                                            for value, label in MAPBOX_STYLES.items()
                                        ],
                                        value=DEFAULT_MAPBOX_STYLE,
                                        inline=True,
                                        **get_dash_persistence_kwargs(persistence_id=True)
                                    )
                                ]),
                                width='auto',
                            )
                        ],
                        justify='between',
                        align='center',
                        style={'margin-top': '15px'},
                    ),
                    dbc.Row(
                        dbc.Col(
                            #id='selected-stations-div',
                            children=[
                                 html.P('Selected stations (you can refine your selection here)',
                                        style={'font-weight': 'bold'}),
                                 dcc.Dropdown(
                                     id=SELECTED_STATIONS_DROPDOWN_ID,
                                     multi=True,
                                     clearable=False,
                                     #**get_dash_persistence_kwargs(persistence_id=True)
                                 ),
                            ],
                        ),
                        style={'margin-top': '15px'},
                    ),
                ]),
            ]
        )
    )
