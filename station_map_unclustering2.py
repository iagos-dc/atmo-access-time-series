import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import xarray as xr

from dash import html, dcc, Dash, callback, Input, Output, State, Patch
import dash
import dash_bootstrap_components as dbc

import utils.stations_map
from utils.charts import IAGOS_COLOR_HEX, rgb_to_rgba, CATEGORY_ORDER, \
    COLOR_CATEGORY_ORDER, ICOS_COLOR_HEX, ACTRIS_COLOR_HEX
import data_access


ZOOM_STORE_ID = 'zoom-store'

APPLY_UNCLUSTERING_SWITCH_ID = 'apply-unclustering-switch'
DIAM_COEFF_INPUT_ID = 'diam-coeff-input' # 3
N_STEPS_INPUT_ID = 'n-steps-input' # 3
FRACTION_OF_DIAM_IN_ONE_JUMP_INPUT_ID = 'fraction-of-diam-in-one-jump-input' # 1
REPULSION_POWER_INPUT_ID = 'repulsion-power-input' # 2
ZOOM_OUTPUT_ID = 'zoom-output'

STATIONS_MAP_ID = 'stations-map'

DEFAULT_MAP_ZOOM = 2

DEFAULT_STATIONS_SIZE = 7
SELECTED_STATIONS_OPACITY = 1.
SELECTED_STATIONS_SIZE = 10
UNSELECTED_STATIONS_OPACITY = 0.5
UNSELECTED_STATIONS_SIZE = 4

MAPBOX_STYLES = {
    'open-street-map': 'open street map',
    'carto-positron': 'carto positron',
}
DEFAULT_MAPBOX_STYLE = 'carto-positron'


stations = data_access.get_stations()


def get_stations_map():
    """
    Provide a Dash component containing a map with stations
    See: https://dash.plotly.com/dash-core-components/graph
    :return: dash.dcc.Graph object
    """
    _stations = stations.copy()
    _stations['marker_size'] = DEFAULT_STATIONS_SIZE
    _stations['lon_displaced'], _stations['lat_displaced'] = utils.stations_map.uncluster(
        _stations['lon_3857_displaced'],
        _stations['lat_3857_displaced'],
        _stations['lon_3857'],
        _stations['lat_3857'],
        DEFAULT_MAP_ZOOM
    )

    fig = px.scatter_mapbox(
        _stations,
        lon="lon_displaced", lat="lat_displaced", color='RI',
        hover_name="long_name",
        hover_data={
            'RI': True,
            'longitude': ':.2f',
            'latitude': ':.2f',
            'ground elevation': _stations['ground_elevation'].round(0).fillna('N/A').to_list(),
            'lon_displaced': False,
            'lat_displaced': False,
            'lon_3857': False,
            'lat_3857': False,
            'marker_size': False
        },
        custom_data=['idx'],
        #text=_stations['marker_size'].astype(str).values,  # it seems text labels still do not work!!! see:
        # https://github.com/mapbox/mapbox-gl-js/issues/4808
        # https://github.com/plotly/plotly.js/pull/6652
        # https://github.com/plotly/plotly.js/issues/4110
        size=_stations['marker_size'],
        size_max=DEFAULT_STATIONS_SIZE,
        category_orders={'RI': CATEGORY_ORDER},
        color_discrete_sequence=COLOR_CATEGORY_ORDER,
        zoom=DEFAULT_MAP_ZOOM,
        # width=1200, height=700,
        center={'lon': 10, 'lat': 55},
        title='Stations map',
    )

    _vectors_lon, _vectors_lat = utils.stations_map.get_displacement_vectors(_stations, DEFAULT_MAP_ZOOM)
    vectors_go = go.Scattermapbox(
        mode='lines',
        lon=_vectors_lon, lat=_vectors_lat,
        connectgaps=False,
        showlegend=False,
        hoverinfo='skip',
        line={'color': 'rgba(0,0,0,1)', 'width': 2},
        #line={'color': 'rgba(0,0,0,0.3)', 'width': 2},
    )
    fig.add_trace(vectors_go)
    *_station_traces, _displacement_vectors = fig.data
    fig.data = _displacement_vectors, *_station_traces

    regions = _stations[_stations['is_region']]
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
        fillcolor=rgb_to_rgba(IAGOS_COLOR_HEX, 0.05),  # IAGOS_COLOR_HEX as rgba with opacity=0.05
        lon=regions_lon,
        lat=regions_lat,
        marker={'color': IAGOS_COLOR_HEX},
        name='IAGOS',
        legendgroup='IAGOS',
    ))

    fig.update_traces(
        selected={'marker_opacity': SELECTED_STATIONS_OPACITY},
        unselected={'marker_opacity': UNSELECTED_STATIONS_OPACITY},
        marker_sizemode='area',
    )

    fig.update_layout(
        mapbox_style=DEFAULT_MAPBOX_STYLE,
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

    return stations_map


@callback(
    Output(STATIONS_MAP_ID, 'figure'),
    Output(ZOOM_OUTPUT_ID, 'children'),
    Output(ZOOM_STORE_ID, 'data'),
    Input(APPLY_UNCLUSTERING_SWITCH_ID, 'value'),
    Input(DIAM_COEFF_INPUT_ID, 'value'),
    Input(N_STEPS_INPUT_ID, 'value'),  # 2
    Input(STATIONS_MAP_ID, 'relayoutData'),
    State(ZOOM_STORE_ID, 'data')
)
def update_map(
        apply_unclustering,
        diam_coeff,
        n_steps,
        map_relayoutData,
        previous_zoom,
):
    zoom = map_relayoutData.get('mapbox.zoom', previous_zoom) if isinstance(map_relayoutData, dict) else previous_zoom
    if apply_unclustering:
        _, _, lon_displaced, lat_displaced = utils.stations_map.uncluster(stations['lon_3857'], stations['lat_3857'], zoom)
    else:
        lon_displaced, lat_displaced = stations['longitude'].values, stations['latitude'].values

    patched_fig = Patch()
    #opacity = pd.Series(UNSELECTED_STATIONS_OPACITY, index=stations.index)
    #size = pd.Series(UNSELECTED_STATIONS_SIZE, index=stations.index)

    df = pd.DataFrame({
        'RI': stations['RI'],
        'longitude': stations['longitude'],
        'latitude': stations['latitude'],
        'lon_displaced': lon_displaced,
        'lat_displaced': lat_displaced,
        #'opacity': opacity,
        #'size': size
    })
    for i, c in enumerate(CATEGORY_ORDER, start=1):
        df_for_c = df[df['RI'] == c]
        #opacity_for_c = df_for_c['opacity']
        #size_for_c = df_for_c['size']
        lon_displaced_for_c = df_for_c['lon_displaced']
        lat_displaced_for_c = df_for_c['lat_displaced']
        patched_fig['data'][i]['lon'] = lon_displaced_for_c
        patched_fig['data'][i]['lat'] = lat_displaced_for_c
        #patched_fig['data'][i]['marker']['opacity'] = opacity_for_c.values
        #patched_fig['data'][i]['marker']['size'] = size_for_c.values
        #patched_fig['data'][i]['selectedpoints'] = None

    i = 0
    patched_fig['data'][i]['lon'], patched_fig['data'][i]['lat'] = utils.stations_map.get_displacement_vectors(df, zoom)

    return patched_fig, str(zoom), zoom


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
    #prevent_initial_callbacks='initial_duplicate',
)


server = app.server


apply_unclustering_switch = dbc.Switch(
    id=APPLY_UNCLUSTERING_SWITCH_ID,
    label='Apply unclustering',
    style={'margin-top': '10px'},
    value=True,
)

input_params = dbc.Form(children=[
    dbc.Row([
        dbc.Label('Repulsion range as a multiple of the marker radius', width=4),
        dbc.Col(
            dcc.Slider(
                id=DIAM_COEFF_INPUT_ID,
                disabled=False,
                min=0.5, max=2, value=1.4, step=0.1,
                persistence=True, persistence_type='session',
            ),
            width=8
        )
    ]),
    dbc.Row([
        dbc.Label('Number of iterations in the repulsion algorithm', width=4),
        dbc.Col(
            dcc.Slider(
                id=N_STEPS_INPUT_ID,
                min=1, max=5, value=2, step=1,
                persistence=True, persistence_type='session',
            ),
            width=8
        )
    ]),
    dbc.Row([
        dbc.Label('Limit displacement in one iteration to this fraction of the marker diameter', width=4),
        dbc.Col(
            dcc.Slider(
                id=FRACTION_OF_DIAM_IN_ONE_JUMP_INPUT_ID,
                disabled=True,
                min=0.1, max=2, value=1,
                persistence=True, persistence_type='session',
            ),
            width=8
        )
    ]),
    dbc.Row([
        dbc.Label('Repulsion force is proportional to (1/distance)^d, d = ', width=4),
        dbc.Col(
            dcc.Slider(
                id=REPULSION_POWER_INPUT_ID,
                disabled=True,
                min=1, max=4, value=3, step=1,
                persistence=True, persistence_type='session',
            ),
            width=8
        )
    ]),
    dbc.Row([
        dbc.Label('Zoom:', width=4),
        dbc.Col(
            html.Div(
                id=ZOOM_OUTPUT_ID,
            ),
            width=8
        )
    ]),
])

#REPULSION_POWER_INPUT_ID = 'repulsion-power-input' # 2



map_tab = dcc.Tab(
    label='2. Search datasets',
    id='SEARCH_DATASETS_TAB_VALUE',
    value='SEARCH_DATASETS_TAB_VALUE',
    children=html.Div(
        style={'margin': '20px'},
        children=[
            html.Div(id='search-datasets-left-panel-div', className='four columns', children=[
                apply_unclustering_switch,
                html.Hr(),
                input_params,
            ]),
            html.Div(id='search-datasets-right-panel-div', className='eight columns', children=get_stations_map()),
        ]
    )
)


app.layout = html.Div(
    id='app-container-div',
    style={'margin': '30px', 'padding-bottom': '50px'},
    children=[
        html.Div(
            id='heading-div',
            className='twelve columns',
            children=[
                map_tab,
            ]
        ),
        dcc.Store(id=ZOOM_STORE_ID, storage_type='session', data=DEFAULT_MAP_ZOOM),
    ]
)
app.title = 'Station map unclustering test'

# Begin of callback definitions and their helper routines.
# See: https://dash.plotly.com/basic-callbacks
# for a basic tutorial and
# https://dash.plotly.com/  -->  Dash Callback in left menu
# for more detailed documentation


# Launch the Dash application in development mode
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
