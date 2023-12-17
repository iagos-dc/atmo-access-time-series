import datetime
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import xarray as xr

from dash import html, dcc, Dash, callback, Input, Output, State, Patch
import dash
import dash_bootstrap_components as dbc

from pyproj import Transformer

from app_tabs.common.data import stations
from utils.charts import IAGOS_COLOR_HEX, rgb_to_rgba, CATEGORY_ORDER, \
    COLOR_CATEGORY_ORDER


_gcs_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
_3857_to_gcs = Transformer.from_crs(3857, 4326, always_xy=True)

ZOOM_STORE_ID = 'zoom-store'

APPLY_UNCLUSTERING_SWITCH_ID = 'apply-unclustering-switch'
DIAM_COEFF_INPUT_ID = 'diam-coeff-input' # 3
N_STEPS_INPUT_ID = 'n-steps-input' # 3
FRACTION_OF_DIAM_IN_ONE_JUMP_INPUT_ID = 'fraction-of-diam-in-one-jump-input' # 1
REPULSION_POWER_INPUT_ID = 'repulsion-power-input' # 2
ZOOM_OUTPUT_ID = 'zoom-output'

STATIONS_MAP_ID = 'stations-map'

DEFAULT_MAP_ZOOM = 2

SELECTED_STATIONS_OPACITY = 1.
SELECTED_STATIONS_SIZE = 10
UNSELECTED_STATIONS_OPACITY = 0.5
UNSELECTED_STATIONS_SIZE = 4

MAPBOX_STYLES = {
    'open-street-map': 'open street map',
    'carto-positron': 'carto positron',
}
DEFAULT_MAPBOX_STYLE = 'carto-positron'


stations['lon_3857'], stations['lat_3857'] = _gcs_to_3857.transform(stations['longitude'], stations['latitude'])


def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.

    See: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlon/2.0)**2 * np.cos(lat2) * np.cos(lat1) + np.sin(dlat/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def haversine_np2(x1, x2):
    x1, x2 = map(np.radians, [x1, x2])
    lat1, lat2 = x1[1], x2[1]
    dlon, dlat = x2 - x1
    a = np.sin(dlon/2.0)**2 * np.cos(lat2) * np.cos(lat1) + np.sin(dlat/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km



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
        category_orders={'RI': CATEGORY_ORDER},
        color_discrete_sequence=COLOR_CATEGORY_ORDER,
        zoom=DEFAULT_MAP_ZOOM,
        # width=1200, height=700,
        center={'lon': 10, 'lat': 55},
        title='Stations map',
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


lon_3857 = stations['lon_3857'].values
lat_3857 = stations['lat_3857'].values
x_ = np.stack([lon_3857, lat_3857], axis=0)
x0 = x_ + np.random.normal(scale=1, size=x_.shape)  # displace by ca. 10m max


def zoom_to_marker_radius_old(zoom):
    if zoom < 2:
        diam_coeff = 1
    elif zoom > 6:
        diam_coeff = 2
    else:
        diam_coeff = 1 + (zoom - 2) / 4
    return diam_coeff * 2**(2 - zoom) * 4e4


def zoom_to_marker_radius(zoom):
    diam_coeff = min(4 / 3 * np.log(2 + zoom), 2.5)
    return diam_coeff * 2**(2 - zoom) * 4e4


def vectors_between_pairs_of_points(x):
    x_ = np.expand_dims(x, axis=-1)
    _x = np.expand_dims(x, axis=-2)
    return _x - x_


def H_old(x, a, d=2):
    xx = vectors_between_pairs_of_points(x)
    d2 = np.square(xx).sum(axis=-3, keepdims=True)
    repulsion = xx / np.power(d2, (d + 1) / 2)
    np.fill_diagonal(repulsion[..., 0, :, :], 0)
    np.fill_diagonal(repulsion[..., 1, :, :], 0)
    return -a**(d + 1) * repulsion.sum(axis=-1)


def H(x, marker_radius, rng):
    xx = vectors_between_pairs_of_points(x)
    d2 = np.square(xx).sum(axis=-3, keepdims=True)
    d = np.sqrt(d2)
    # repulsion = np.maximum(a - d, 0) * xx / d
    repulsion = np.maximum(marker_radius - d / rng, 0) * xx / d
    np.fill_diagonal(repulsion[..., 0, :, :], 0)
    np.fill_diagonal(repulsion[..., 1, :, :], 0)
    return -repulsion.sum(axis=-1)


def repulse_old(x0, diam, n_steps, jump=0.1, d=2):
    x1 = x0
    for i in range(n_steps):
        h = H_old(x1, diam)
        dh = np.sqrt(np.square(h).sum(axis=-2, keepdims=True))
        h = h / np.maximum(1 / jump, dh / diam)
        x1 = x1 + h
        dist = np.sqrt(np.square(x1 - x0).sum(axis=-2, keepdims=True))
        x1 = x0 + (x1 - x0) / np.maximum(1, dist / diam)
    return x1


def repulse(x0, marker_radius, rng, n_steps):
    x1 = x0
    for i in range(n_steps):
        x1 = x1 + H(x1, marker_radius, rng)
    return x1


@callback(
    Output(STATIONS_MAP_ID, 'figure'),
    Output(ZOOM_OUTPUT_ID, 'children'),
    Output(ZOOM_STORE_ID, 'data'),
    Input(APPLY_UNCLUSTERING_SWITCH_ID, 'value'),
    Input(DIAM_COEFF_INPUT_ID, 'value'),
    Input(N_STEPS_INPUT_ID, 'value'),  # 2
    Input(FRACTION_OF_DIAM_IN_ONE_JUMP_INPUT_ID, 'value'),  # 1
    Input(REPULSION_POWER_INPUT_ID, 'value'),  # 3
    Input(STATIONS_MAP_ID, 'relayoutData'),
    State(ZOOM_STORE_ID, 'data')
)
def update_map(
        apply_unclustering,
        diam_coeff,
        n_steps,
        fraction_of_diam_in_one_jump,
        repulsion_power,
        map_relayoutData,
        previous_zoom,
):
    zoom = map_relayoutData.get('mapbox.zoom', previous_zoom) if isinstance(map_relayoutData, dict) else previous_zoom

    if apply_unclustering:
        #x1 = repulse(x0, diam=diam_coeff * zoom_to_diam(zoom), n_steps=n_steps, jump=fraction_of_diam_in_one_jump / 2, d=repulsion_power)
        x1 = repulse(x0, marker_radius=zoom_to_marker_radius(zoom), rng=diam_coeff, n_steps=n_steps)
        lon_displaced, lat_displaced = _3857_to_gcs.transform(x1[0], x1[1])
        stations['lon_displaced'] = pd.Series(lon_displaced, index=stations.index)
        stations['lat_displaced'] = pd.Series(lat_displaced, index=stations.index)
        lon_column = 'lon_displaced'
        lat_column = 'lat_displaced'
    else:
        lon_column = 'longitude'
        lat_column = 'latitude'

    patched_fig = Patch()

    for i, c in enumerate(CATEGORY_ORDER):
        df_for_c = stations[stations['RI'] == c]
        patched_fig['data'][i]['lon'] = df_for_c[lon_column].values
        patched_fig['data'][i]['lat'] = df_for_c[lat_column].values

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
