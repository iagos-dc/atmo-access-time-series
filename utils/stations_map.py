import numpy as np
import pandas as pd
import sklearn.cluster
from plotly import express as px, graph_objects as go
from pyproj import Transformer

from app_tabs.common.data import stations
from utils.charts import CATEGORY_ORDER, COLOR_CATEGORY_ORDER, rgb_to_rgba, IAGOS_COLOR_HEX

from data_access import get_stations


DEFAULT_STATIONS_SIZE = 7
SELECTED_STATIONS_SIZE = 8
UNSELECTED_STATIONS_SIZE = 6
SELECTED_STATIONS_OPACITY = 1.
UNSELECTED_STATIONS_OPACITY = 0.5

_DIAM_COEFF = 1.4
_N_STEPS = 2
_gcs_to_3857 = Transformer.from_crs(4326, 3857, always_xy=True)
_3857_to_gcs = Transformer.from_crs(3857, 4326, always_xy=True)


# 'lon_3857', 'lat_3857'.
#    The last two columns are the Mercator coordinates and serve for the stations unclustering algorithm.
# 'lon_3857': 918597.5036487236,
# 'lat_3857': 8049397.769844955

all_stations_df = get_stations()
_lon_3857, _lat_3857 = _gcs_to_3857.transform(all_stations_df['longitude'], all_stations_df['latitude'])
_lon_3857 += np.random.normal(scale=1, size=_lon_3857.shape)
_lat_3857 += np.random.normal(scale=1, size=_lat_3857.shape)
all_stations_df['lon_3857'] = _lon_3857
all_stations_df['lat_3857'] = _lat_3857


def _zoom_to_marker_radius(zoom):
    diam_coeff = min(4 / 3 * np.log(2 + zoom), 2.5)
    return diam_coeff * 2 ** (2 - zoom) * 4e4


def _vectors_between_pairs_of_points(x):
    x_ = np.expand_dims(x, axis=-1)
    _x = np.expand_dims(x, axis=-2)
    return _x - x_


def _H(x, marker_radius, rng):
    xx = _vectors_between_pairs_of_points(x)
    d2 = np.square(xx).sum(axis=-3, keepdims=True)
    d = np.sqrt(d2)
    repulsion = np.maximum(marker_radius - d / rng, 0) * xx / d
    np.fill_diagonal(repulsion[..., 0, :, :], 0)
    np.fill_diagonal(repulsion[..., 1, :, :], 0)
    return -repulsion.sum(axis=-1)


def _repulse(x0, marker_radius, rng, n_steps):
    x1 = x0
    for i in range(n_steps):
        x1 = x1 + _H(x1, marker_radius, rng)
    return x1


def uncluster(lon_3857, lat_3857, zoom):
    x0 = np.stack([lon_3857, lat_3857], axis=0)
    x1 = _repulse(x0, marker_radius=_zoom_to_marker_radius(zoom), rng=_DIAM_COEFF, n_steps=_N_STEPS)
    lon_3857_displaced, lat_3857_displaced = x1
    lon_displaced, lat_displaced = _3857_to_gcs.transform(lon_3857_displaced, lat_3857_displaced)
    if isinstance(lon_3857, pd.Series):
        index = lon_3857.index
    elif isinstance(lat_3857, pd.Series):
        index = lat_3857.index
    else:
        index = None
    if index is not None:
        lon_displaced = pd.Series(lon_displaced, index=index)
        lat_displaced = pd.Series(lat_displaced, index=index)
        lon_3857_displaced = pd.Series(lon_3857_displaced, index=index)
        lat_3857_displaced = pd.Series(lat_3857_displaced, index=index)
    return lon_3857_displaced, lat_3857_displaced, lon_displaced, lat_displaced


def uncluster2(stations, zoom):
    if zoom <= 3:
        _stations = cluster(stations, zoom)
        _stations['marker_size'] = 14 # 2 * DEFAULT_STATIONS_SIZE
    else:
        _stations = stations.copy()
        _stations['marker_size'] = 7 # DEFAULT_STATIONS_SIZE

    _, _, _stations['lon_displaced'], _stations['lat_displaced'] = uncluster(
        _stations['lon_3857'],
        _stations['lat_3857'],
        zoom
    )
    return _stations


def get_displacement_vectors(df, zoom):
    d = _zoom_to_marker_radius(zoom)
    not_a_small_displacement = np.sqrt(
        np.square(df['lon_3857_displaced'] - df['lon_3857']) +
        np.square(df['lat_3857_displaced'] - df['lat_3857'])
    ) > 0.1 * d
    df = df[not_a_small_displacement]
    n = len(df)
    lon = np.full(3 * n, np.nan)
    lat = np.full(3 * n, np.nan)
    lon[::3] = df['longitude'].values
    lon[1::3] = df['lon_displaced'].values
    lat[::3] = df['latitude'].values
    lat[1::3] = df['lat_displaced'].values
    return lon, lat


def cluster(stations, zoom):
    clusters = []
    for ri, stations_for_ri in stations.groupby('RI'):
        kmeans_algo = sklearn.cluster.KMeans(n_clusters=int(1 + 5 * 2**zoom), random_state=0)
        lon_lat = stations_for_ri[['lon_3857', 'lat_3857']].values
        kmeans_model = kmeans_algo.fit(lon_lat)
        clusters_lon, clusters_lat = kmeans_model.cluster_centers_.T
        clusters_for_ri = pd.DataFrame({'lon_3857': clusters_lon, 'lat_3857': clusters_lat})

        clusters_idx = kmeans_model.predict(lon_lat)
        stations_for_ri['cluster_idx'] = clusters_idx
        station_names = stations_for_ri.groupby('cluster_idx')['long_name'].agg(list)
        clusters_for_ri['long_name'] = station_names
        clusters_for_ri['RI'] = ri

        clusters.append(clusters_for_ri)
    return pd.concat(clusters, axis='index', ignore_index=True)


def get_stations_map(zoom):
    """
    Provide a Dash component containing a map with stations
    See: https://dash.plotly.com/dash-core-components/graph
    :return: dash.dcc.Graph object
    """
    _stations = stations.copy()
    _stations['marker_size'] = DEFAULT_STATIONS_SIZE
    _stations['lon_3857_displaced'], _stations['lat_3857_displaced'], _stations['lon_displaced'], _stations['lat_displaced'] = uncluster(
        _stations['lon_3857'],
        _stations['lat_3857'],
        zoom
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
        size='marker_size',
        size_max=DEFAULT_STATIONS_SIZE,
        category_orders={'RI': CATEGORY_ORDER},
        color_discrete_sequence=COLOR_CATEGORY_ORDER,
    )

    _vectors_lon, _vectors_lat = get_displacement_vectors(_stations, zoom)
    # add the trace with displacement vectors
    vectors_go = go.Scattermapbox(
        mode='lines',
        lon=_vectors_lon, lat=_vectors_lat,
        connectgaps=False,
        showlegend=False,
        hoverinfo='skip',
        line={'color': 'rgba(0,0,0,1)', 'width': 1},
    )
    fig.add_trace(vectors_go)
    # add the trace with the vectors' endpoint pointing to the stations' position
    stations_positions_go = go.Scattermapbox(
        mode='markers',
        lon=_vectors_lon[::3], lat=_vectors_lat[::3],
        showlegend=False,
        hoverinfo='skip',
        marker={'color': 'rgba(0,0,0,1)', 'size': [4, ] * len(_vectors_lon[::3])},
    )
    fig.add_trace(stations_positions_go)
    # Swap traces corresponding to RIs' stations and the displacement vectors and points in order to have the stations'
    # markers on the top of the two other traces.
    *_station_traces, _displacement_vectors, _stations_positions = fig.data
    fig.data = _displacement_vectors, _stations_positions, *_station_traces

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
        #marker_sizemode='area',
    )

    fig.update_layout(
        legend={
            'groupclick': 'toggleitem',
            'traceorder': 'grouped',
            'orientation': 'v',
        }
    )

    return fig
