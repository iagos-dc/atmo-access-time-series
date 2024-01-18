import dash_bootstrap_components as dbc
from dash import dcc, html

import utils.stations_map
from app_tabs.common.layout import SEARCH_DATASETS_TAB_VALUE, get_next_button, std_variables
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

# 'value' contains a number (or None)

VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'variables-checklist-all-none-switch'

SELECTED_STATIONS_DROPDOWN_ID = 'selected-stations-dropdown'
# 'options' contains a list of dictionaries {'label' -> station label, 'value' -> index of the station in the global dataframe stations (see below)}
# 'value' contains a list of indices of stations selected using the dropdown

SEARCH_DATASETS_BUTTON_ID = 'search-datasets-button'
# 'n_click' contains a number of clicks at the button

SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID = 'search-datasets-reset-stations-button'

MAP_BACKGROUND_RADIO_ID = 'map-background-radio'

MAPBOX_STYLES = {
    'open-street-map': 'open street map',
    'carto-positron': 'carto positron',
}
DEFAULT_MAPBOX_STYLE = 'carto-positron'


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

    return [
        dcc.Store(id=MAP_ZOOM_STORE_ID, data=DEFAULT_MAP_ZOOM),
        stations_map
    ]


def get_search_datasets_tab():
    variable_selection_card = dbc.Card([
        dbc.CardHeader(
            'Select variables',
            style={'font-weight': 'bold'},
        ),
        dbc.CardBody([
            dbc.Switch(
                id=VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID,
                label='Select all / none',
                #style={'margin-top': '10px'},
                value=True,
            ),
            get_variables_checklist(),
        ])
    ])

    station_selection_card = dbc.Card([
        dbc.CardHeader(
            'Select stations',
            style={'font-weight': 'bold'},
        ),
        dbc.CardBody([
            dbc.Row(
                dbc.Col(
                    get_stations_dash_component(),
                ),
            ),
            dbc.Row(
                html.Div(
                    [
                        html.Div(
                            dbc.Button(
                                id=SEARCH_DATASETS_RESET_STATIONS_BUTTON_ID,
                                color='danger',
                                type='submit',
                                style={'font-weight': 'bold', 'margin-bottom': '10px'},
                                children='Clear',
                            ),
                        ),
                        html.Div(
                            dbc.InputGroup(
                                [
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
                                ],
                                size='lg',
                                style={
                                    'display': 'flex',
                                    'align-items': 'center',
                                    'border': '1px solid lightgrey',
                                    'border-radius': '5px'
                                }
                            ),
                        ),
                    ],
                    style={
                        'display': 'flex',
                        'justify-content': 'space-between',
                        'align-items': 'center',
                        'margin-top': '15px'
                    },
                ),
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
                style={'margin-top': '10px'},
            ),
        ]),
    ])

    return dbc.Tab(
        label='2. Search datasets',
        id=SEARCH_DATASETS_TAB_VALUE,
        tab_id=SEARCH_DATASETS_TAB_VALUE,
        children=html.Div(
            style={'margin-top': '5px', 'margin-left': '20px', 'margin-right': '20px'},
            children=[
                dbc.Row(
                    dbc.Col(
                        children=html.Div(get_next_button(SEARCH_DATASETS_BUTTON_ID)),
                        width='auto',
                    ),
                    justify='end',
                    style={'margin-bottom': '10px'},
                ),
                dbc.Row([
                    dbc.Col(
                        width=3,
                        children=variable_selection_card
                    ),
                    dbc.Col(
                        width=9,
                        children=station_selection_card
                    )
                ])
            ]
        )
    )
