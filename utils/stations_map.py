import numpy as np
import pandas as pd
import sklearn.cluster
from plotly import graph_objects as go
from dash import Patch
from pyproj import Transformer

from app_tabs.common.data import stations
from utils.charts import CATEGORY_ORDER, COLOR_CATEGORY_ORDER, COLOR_BY_CATEGORY, rgb_to_rgba, IAGOS_COLOR_HEX, change_rgb_brightness2

from data_access import get_stations


DEFAULT_STATIONS_SIZE = 10
SELECTED_STATIONS_SIZE = 11
UNSELECTED_STATIONS_SIZE = 9
SELECTED_STATIONS_OPACITY = 1.
UNSELECTED_STATIONS_OPACITY = 0.75

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


def get_stations_with_displacement_coords(stations, zoom):
    _stations = stations.copy()
    _stations['lon_3857_displaced'], _stations['lat_3857_displaced'], _stations['lon_displaced'], _stations['lat_displaced'] = uncluster(
        _stations['lon_3857'],
        _stations['lat_3857'],
        zoom
    )
    return _stations


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


def get_station_and_displacement_vector_and_ori_position_traces(df, color, zoom, customdata_columns, hovertemplate=None, trace_kwargs=None, marker_kwargs=None):
    if trace_kwargs is None:
        trace_kwargs = {}
    if marker_kwargs is None:
        marker_kwargs = {}

    station_trace = go.Scattermapbox(
        lon=df['lon_displaced'].values,
        lat=df['lat_displaced'].values,
        customdata=df[customdata_columns].values,
        hovertemplate=hovertemplate,
        hovertext=df['long_name'],
        hoverinfo=None,
        marker={
            'color': color,
            'size': [DEFAULT_STATIONS_SIZE] * len(df),
            'opacity': 1,
            **marker_kwargs
        },
        mode='markers',
        showlegend=True,
        **trace_kwargs
    )

    _vectors_lon, _vectors_lat = get_displacement_vectors(df, zoom)

    if len(_vectors_lon) == 0:
        # this is a workaround of a fact that plotly does not like a subsequent update of an initially empty trace
        # without the two lines below, an update of the trace related to displacement of regions' markers behaved badly;
        # NB: must be lists, numpy arrays will not work...
        _vectors_lon = [np.nan, np.nan, np.nan]
        _vectors_lat = [np.nan, np.nan, np.nan]

    displacement_vector_trace = go.Scattermapbox(
        mode='lines',
        lon=_vectors_lon, lat=_vectors_lat,
        connectgaps=False,
        showlegend=False,
        hoverinfo='skip',
        line={'color': 'rgba(0,0,0,1)', 'width': 1},
        **trace_kwargs
    )
    station_original_position_trace = go.Scattermapbox(
        mode='markers',
        lon=_vectors_lon[::3], lat=_vectors_lat[::3],
        showlegend=False,
        hoverinfo='skip',
        marker={'color': 'rgba(0,0,0,1)', 'size': [4, ] * len(_vectors_lon[::3])},
        **trace_kwargs
    )
    return station_trace, displacement_vector_trace, station_original_position_trace


def update_fig_station_and_displacement_vector_and_ori_position_traces(
        patched_fig,
        df,
        zoom,
        station_trace_index,
        displacement_vector_index,
        ori_position_trace_index
):
    # update stations
    patched_fig['data'][station_trace_index]['lon'] = df['lon_displaced'].values
    patched_fig['data'][station_trace_index]['lat'] = df['lat_displaced'].values

    patched_fig['data'][station_trace_index]['marker']['opacity'] = df['_opacity'].values
    patched_fig['data'][station_trace_index]['marker']['size'] = df['_size'].values
    patched_fig['data'][station_trace_index]['selectedpoints'] = None

    _vectors_lon, _vectors_lat = get_displacement_vectors(df, zoom)

    # update displacement vectors
    patched_fig['data'][displacement_vector_index]['lon'] = _vectors_lon
    patched_fig['data'][displacement_vector_index]['lat'] = _vectors_lat

    # update original station positions
    patched_fig['data'][ori_position_trace_index]['lon'] = _vectors_lon[::3]
    patched_fig['data'][ori_position_trace_index]['lat'] = _vectors_lat[::3]

    return patched_fig


def get_stations_map(zoom):
    """
    Provide a Dash component containing a map with stations
    See: https://dash.plotly.com/dash-core-components/graph
    :return: dash.dcc.Graph object
    """
    _stations = get_stations_with_displacement_coords(stations, zoom)

    _ground_stations = _stations[~_stations['is_region']]
    _regions = _stations[_stations['is_region']]

    fig = go.Figure()

    station_traces = {}
    displacement_vector_traces = {}
    station_original_position_traces = {}

    # ground stations
    station_customdata_columns = ['idx', 'RI', 'latitude', 'longitude', 'ground_elevation']
    station_hovertemplate = '<b>%{hovertext}</b><br><br>' \
                            'RI=%{customdata[1]}<br>' \
                            'latitude=%{customdata[2]:.2f}<br>' \
                            'longitude=%{customdata[3]:.2f}<br>' \
                            'ground elevation=%{customdata[4]:.0f}' \
                            '<extra></extra>'
    for ri, _stations_for_ri in _ground_stations.groupby('RI'):
        color = COLOR_BY_CATEGORY[ri]
        trace_kwargs = dict(
            legendgroup=ri,
            name=ri,
        )
        station_trace, displacement_vector_trace, station_original_position_trace = \
            get_station_and_displacement_vector_and_ori_position_traces(
                _stations_for_ri,
                color,
                zoom,
                station_customdata_columns,
                hovertemplate=station_hovertemplate,
                trace_kwargs=trace_kwargs
            )

        station_traces[ri] = station_trace
        displacement_vector_traces[ri] = displacement_vector_trace
        station_original_position_traces[ri] = station_original_position_trace

    # regions
    region_customdata_columns = ['idx', 'RI', 'longitude', 'latitude']
    region_hovertemplate = '<b>%{hovertext}</b><br><br>' \
                            'RI=%{customdata[1]}<br>' \
                            'latitude=%{customdata[2]:.2f}<br>' \
                            'longitude=%{customdata[3]:.2f}<br>' \
                            '<extra></extra>'
    color = change_rgb_brightness2(COLOR_BY_CATEGORY['IAGOS'], 20)
    regions_trace_kwargs = dict(
        name='IAGOS region marker<br>(click to select)',
        legendgroup='IAGOS-regions',
        legendgrouptitle_text='Regional samples',
    )
    station_trace, displacement_vector_trace, station_original_position_trace = \
        get_station_and_displacement_vector_and_ori_position_traces(
            _regions,
            color,
            zoom,
            region_customdata_columns,
            hovertemplate=region_hovertemplate,
            trace_kwargs=regions_trace_kwargs,
            # marker_kwargs={'symbol': 'airport'}  # it does not work as intended! (plotly.js bug? would have to use non-free mapbox vector layers)
        )

    station_traces['IAGOS_region'] = station_trace
    displacement_vector_traces['IAGOS_region'] = displacement_vector_trace
    station_original_position_traces['IAGOS_region'] = station_original_position_trace

    # add the trace with displacement vectors
    fig.add_traces([displacement_vector_traces[ri] for ri in CATEGORY_ORDER + ['IAGOS_region']])
    # add the trace with the vectors' endpoint pointing to the stations' position
    fig.add_traces([station_original_position_traces[ri] for ri in CATEGORY_ORDER + ['IAGOS_region']])
    # add the trace with stations / regions
    #fig.add_traces([station_traces[ri] for ri in CATEGORY_ORDER + ['IAGOS_region']])
    fig.add_traces([station_traces[ri] for ri in CATEGORY_ORDER])

    regions_lon = []
    regions_lat = []
    for lon_min, lon_max, lat_min, lat_max in zip(_regions['longitude_min'], _regions['longitude_max'], _regions['latitude_min'], _regions['latitude_max']):
        if len(regions_lon) > 0:
            regions_lon.append(None)
            regions_lat.append(None)
        regions_lon.extend([lon_min, lon_min, lon_max, lon_max, lon_min])
        regions_lat.extend([lat_min, lat_max, lat_max, lat_min, lat_min])

    fig.add_trace(go.Scattermapbox(
        mode="lines",
        connectgaps=False,
        fill="toself",
        fillcolor=rgb_to_rgba(IAGOS_COLOR_HEX, 0.05),  # IAGOS_COLOR_HEX as rgba with opacity=0.05
        lon=regions_lon,
        lat=regions_lat,
        #marker={'color': IAGOS_COLOR_HEX},
        line={'color': IAGOS_COLOR_HEX, 'width': 1},
        showlegend=True,
        name='IAGOS region boundary',
        legendgroup='IAGOS-regions',
        #legendgrouptitle_text='Regional samples at 200 hPa',
    ))

    fig.add_trace(station_traces['IAGOS_region'])

    fig.update_traces(
        selected={'marker_opacity': SELECTED_STATIONS_OPACITY},
        unselected={'marker_opacity': UNSELECTED_STATIONS_OPACITY},
    )

    fig.update_layout(
        legend={
            'title': 'Ground based stations',
            'tracegroupgap': 0,
            #'groupclick': 'toggleitem',
            'traceorder': 'grouped',
            #'orientation': 'v',
            #'xref': 'container',
            #'yref': 'container',
            #'y': 1,
        },
    )

    return fig


def get_stations_map_patch(zoom, size, opacity):
    _stations = get_stations_with_displacement_coords(stations, zoom)
    _stations['_size'] = size
    _stations['_opacity'] = opacity

    _ground_stations = _stations[~_stations['is_region']]
    _regions = _stations[_stations['is_region']]

    patched_fig = Patch()

    # ground stations
    for ri, _stations_for_ri in _ground_stations.groupby('RI'):
        index_offset = CATEGORY_ORDER.index(ri)
        patched_fig = update_fig_station_and_displacement_vector_and_ori_position_traces(
            patched_fig,
            _stations_for_ri,
            zoom,
            station_trace_index=8 + index_offset,
            displacement_vector_index=index_offset,
            ori_position_trace_index=4 + index_offset
        )

    # regions
    patched_fig = update_fig_station_and_displacement_vector_and_ori_position_traces(
        patched_fig,
        _regions,
        zoom,
        station_trace_index=12,
        displacement_vector_index=3,
        ori_position_trace_index=7
    )

    return patched_fig
