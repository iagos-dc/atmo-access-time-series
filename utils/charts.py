import functools
import toolz
import numpy as np
import pandas as pd
import xarray as xr
from plotly import express as px, graph_objects as go
import plotly.colors
import textwrap

from log.log import logger, log_exectime
from data_processing.utils import get_subsampling_mask
from utils import helper
from utils.exception_handler import EmptyFigureException


# Color codes
ACTRIS_COLOR_HEX = '#00adb7'
IAGOS_COLOR_HEX = '#456096'
ICOS_COLOR_HEX = '#ec165c'

CATEGORY_ORDER = ['ACTRIS', 'IAGOS', 'ICOS']
COLOR_BY_CATEGORY = {'ACTRIS': ACTRIS_COLOR_HEX, 'IAGOS': IAGOS_COLOR_HEX, 'ICOS': ICOS_COLOR_HEX}
COLOR_CATEGORY_ORDER = [COLOR_BY_CATEGORY[c] for c in CATEGORY_ORDER]

MAX_WRAP = 20


def _wrap_text(s):
    if len(s) <= MAX_WRAP:
        return s
    else:
        return '<br>'.join(textwrap.wrap(s, MAX_WRAP))


def rgb_to_tuple(rgb):
    rgb_tuple = None
    if isinstance(rgb, tuple):
        rgb_tuple = rgb
    elif isinstance(rgb, str):
        if rgb.startswith('rgb(') and rgb.endswith(')'):
            rgb_tuple = plotly.colors.unlabel_rgb(rgb)
        elif rgb.startswith('#'):
            rgb_tuple = plotly.colors.hex_to_rgb(rgb)

    if rgb_tuple is None:
        raise ValueError(f'invalid format of rgb; must be a tuple of 3 ints, a string rgb(R, G, B) or hex; got {rgb}')
    return rgb_tuple


def rgb_to_rgba(rgb, opacity):
    rgb_tuple = rgb_to_tuple(rgb)
    rgba_tuple = rgb_tuple + (opacity, )
    return f'rgba{rgba_tuple}'


def change_rgb_brightness(rgb, brightness_coeff):
    assert -1 <= brightness_coeff <= 1, f'brightness coeff must be between -1 and 1, got={brightness_coeff}'
    rgb_tuple = rgb_to_tuple(rgb)
    #rgb_max = max(rgb_tuple)
    rgb_array = np.array(rgb_tuple)
    if brightness_coeff >= 0:
        rgb_array = rgb_array * (1 + brightness_coeff)# * 255 / rgb_max)
        #rgb_array = rgb_array * (brightness_coeff * 255 / rgb_max)
    else:
        rgb_array = rgb_array * (1 + brightness_coeff)
    new_rgb_tuple = tuple(np.clip(rgb_array.astype(int), 0, 255))
    return f'rgb{new_rgb_tuple}'


def change_rgb_brightness2(rgb, brightness_coeff):
    rgb_tuple = rgb_to_tuple(rgb)
    rgb_array = np.array(rgb_tuple)
    rgb_array = rgb_array + brightness_coeff
    new_rgb_tuple = tuple(np.clip(rgb_array.astype(int), 0, 255))
    return f'rgb{new_rgb_tuple}'



def plotly_scatter(x, y, *args, y_std=None, std_mode=None, std_fill_opacity=0.2, use_GL='auto', **kwargs):
    """
    This a wrapper around plotly.graph_objects.Scatter for the purpose of plotting std as area plot
    TODO: consider NOT masking out isolate points; what about std (as filled area) for isolated points?
    TODO: refactor since the plotly bug mentioned below is already fixed
    seems no longer an issue, and the workaround was removed
    OLD: It workarounds plotly bug: Artifacts on line scatter plot when the first item is None #3959
    OLD: https://github.com/plotly/plotly.py/issues/3959

    :param std_mode: 'fill', 'error_bars' or None;
    """
    def mask_isolated_values(ar):
        ar_isnan = np.isnan(ar).astype('i4')
        isolated_points = np.diff(ar_isnan, n=2, prepend=1, append=1) == 2
        return np.where(~isolated_points, ar, np.nan)

    assert std_mode in ['fill', 'error_bars', None]

    x = np.asanyarray(x)
    y = np.asanyarray(y)
    # y = mask_isolated_values(y)

    # print(f'plotly_scatter: len(x)={len(x)}')

    # TODO: test if we can mix traces of scatter and scattergl type on a single go.Figure
    if use_GL is False or use_GL == 'auto' and (len(x) <= 500 or y_std is not None and std_mode == 'fill'):
        go_Scatter = go.Scatter
        go_scatter_ErrorY = go.scatter.ErrorY
    else:
        # for many points, we use GL version: https://github.com/plotly/plotly.js/issues/741
        # but there is a bug with fill='tonexty': https://github.com/plotly/plotly.py/issues/2018,
        #   https://github.com/plotly/plotly.py/issues/2322, https://github.com/plotly/plotly.js/issues/4017
        # for further improvements see: https://github.com/plotly/plotly.js/issues/6230 (e.g. layout.hovermode = 'x')
        go_Scatter = go.Scattergl
        go_scatter_ErrorY = go.scattergl.ErrorY

    if y_std is not None:
        y_std = np.asanyarray(y_std)
        if std_mode == 'error_bars':
            y_std = np.where(~np.isnan(y), y_std, np.nan)
            kwargs = kwargs.copy()
            kwargs['error_y'] = go_scatter_ErrorY(array=y_std, symmetric=True, type='data')

    y_scatter = go_Scatter(
        x=x,
        y=y,
        *args,
        **kwargs
    )
    if y_std is None or std_mode != 'fill':
        return y_scatter
    else:
        # y_std is not None and std_mode == 'fill'
        y_std = np.asanyarray(y_std)
        # this is a workaround to the plotly bug https://github.com/plotly/plotly.js/issues/2736
        y_lo = mask_isolated_values(y - y_std)
        y_hi = mask_isolated_values(y + y_std)
        y_lo_isnan = np.isnan(y_lo)
        _block_of_values_begins_after_nan = np.nonzero((np.diff(y_lo_isnan.astype('i4')) == -1))[0] + 1
        _block_of_values_ends_before_nan = np.nonzero((np.diff(y_lo_isnan) == 1))[0]
        _idx_to_insert_before = np.concatenate((_block_of_values_begins_after_nan, _block_of_values_ends_before_nan + 1))
        _x_values_to_insert = np.concatenate((x[_block_of_values_begins_after_nan], x[_block_of_values_ends_before_nan]))
        _y_values_to_insert = np.concatenate((y[_block_of_values_begins_after_nan], y[_block_of_values_ends_before_nan]))
        x_std = np.insert(x, _idx_to_insert_before, _x_values_to_insert)
        y_lo = np.insert(y_lo, _idx_to_insert_before, _y_values_to_insert)
        y_hi = np.insert(y_hi, _idx_to_insert_before, _y_values_to_insert)

        y_lo_notnan = ~np.isnan(y_lo)
        x_std = x_std[y_lo_notnan]
        y_lo = y_lo[y_lo_notnan]
        y_hi = y_hi[y_lo_notnan]

        kwargs_lo = kwargs.copy()
        kwargs_lo['mode'] = 'lines'
        kwargs_lo.pop('line_width', None)
        kwargs_lo.setdefault('line', {})
        kwargs_lo['line']['width'] = 0
        kwargs_lo['hoverinfo'] = 'skip'
        kwargs_lo['showlegend'] = False

        y_scatter_lo = plotly_scatter(x_std, y_lo, use_GL=use_GL, *args, **kwargs_lo)

        kwargs_hi = kwargs_lo.copy()
        kwargs_hi['fill'] = 'tonexty'
        color_rgb = None
        if 'marker' in kwargs_hi and isinstance(kwargs_hi['marker'], dict) and 'color' in kwargs_hi['marker']:
            color_rgb = kwargs_hi['marker']['color']
        elif 'marker_color' in kwargs_hi:
            color_rgb = kwargs_hi['marker_color']
        elif 'line' in kwargs_hi and isinstance(kwargs_hi['line'], dict) and 'color' in kwargs_hi['line']:
            color_rgb = kwargs_hi['line']['color']
        elif 'line_color' in kwargs_hi:
            color_rgb = kwargs_hi['line_color']
        if color_rgb is not None:
            kwargs_hi['fillcolor'] = rgb_to_rgba(color_rgb, std_fill_opacity)

        y_scatter_hi = plotly_scatter(x_std, y_hi, use_GL=use_GL, *args, **kwargs_hi)

        return y_scatter, y_scatter_lo, y_scatter_hi


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
        .groupby(['platform_id_RI', 'station_fullname', 'RI'], group_keys=True)\
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
        category_orders={'RI': CATEGORY_ORDER},
        color_discrete_sequence=COLOR_CATEGORY_ORDER,
        height=height
    )

    gantt.update_layout(
        clickmode='event',
        # selectdirection='h',
        legend={
            'title': 'RI',
            'orientation': 'h',
            'yanchor': 'bottom', 'y': 1, 'xanchor': 'left', 'x': 0
        },
        yaxis={
            'title': 'Platform',
            'side': 'right',
            'autorange': 'reversed',
            'tickmode': 'array',
            'tickvals': df['platform_id_RI'],
            'ticktext': df['station_fullname'].map(_wrap_text),
        },
    )
    return gantt


def _get_timeline_by_station_and_vars(datasets_df):
    df = datasets_df\
        .groupby(['platform_id_RI', 'station_fullname', 'var_codes_filtered'])\
        .apply(lambda x: _contiguous_periods(x['time_period_start'], x['time_period_end']))\
        .reset_index()
    df = df.sort_values('platform_id_RI')
    facet_col_wrap = 4
    no_platforms = len(df['platform_id_RI'].unique())
    no_var_codes_filtered = len(df['var_codes_filtered'].unique())
    no_facet_rows = (no_var_codes_filtered + facet_col_wrap - 1) // facet_col_wrap
    height = 100 + max(100, 50 + 25 * no_platforms) * no_facet_rows
    platform_id_RI_var_codes_filtered = df['var_codes_filtered'] + ' : ' + df['platform_id_RI']
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y=platform_id_RI_var_codes_filtered, color='var_codes_filtered',
        hover_name='station_fullname',
        hover_data={'station_fullname': True, 'platform_id_RI': True, 'var_codes_filtered': True, 'datasets': True},
        custom_data=['indices'],
        height=height,
    )
    gantt.update_layout(
        clickmode='event',
        # selectdirection='h',
        legend={
            'title': 'Variable(s)',
            'orientation': 'h',
            'yanchor': 'bottom', 'y': 1, 'xanchor': 'left', 'x': 0
        },
        yaxis={
            'title': 'Platform : Variable(s)',
            'side': 'right',
            'autorange': 'reversed',
            'tickmode': 'array',
            'tickvals': platform_id_RI_var_codes_filtered,
            'ticktext': df['station_fullname'].map(_wrap_text),
        }
    )
    # print(f'gantt={json.loads(gantt.to_json())}')
    return gantt


def get_avail_data_by_var_gantt(ds):
    dfs = []
    for v, da in ds.data_vars.items():
        *v_label, ri = v.split('_')
        v_label = '_'.join(v_label)
        notnull_diff = da.notnull().astype('i2').diff('time')

        s = da['time'].where(notnull_diff == 1, drop=True).values
        if da.isel({'time': 0}).notnull():
            s = np.concatenate((da['time'].values[:1], s))
        e = da['time'].where(notnull_diff == -1, drop=True).values
        if da.isel({'time': -1}).notnull():
            e = np.concatenate((e, da['time'].values[-1:]))

        df = _contiguous_periods(s, e)[['time_period_start', 'time_period_end']]
        df['var_label'] = v_label
        df['RI'] = ri
        df['variable (RI)'] = f'{v_label} ({ri})'
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    # df = df.sort_values('platform_id_RI')
    height = 200 + max(80, 30 + 10 * len(ds.data_vars))
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y='variable (RI)', color='RI',
        hover_name='variable (RI)',
        hover_data={'var_label': True, 'RI': False},
        # custom_data=['indices'],
        category_orders={'RI': CATEGORY_ORDER},
        color_discrete_sequence=COLOR_CATEGORY_ORDER,
        height=height
    )
    gantt.update_layout(
        clickmode='event',
        selectdirection='h',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.04, 'xanchor': 'left', 'x': 0},
    )
    return gantt


@functools.lru_cache()
def colors():
    list_of_rgb_triples = [plotly.colors.hex_to_rgb(hex_color) for hex_color in px.colors.qualitative.Dark24]
    return list_of_rgb_triples


def get_color_mapping(variables):
    if len(variables) > len(colors()):
        raise ValueError(f'too many variables: len(variables)={len(variables)} cannot exceed len(colors())={len(colors())}')
    return dict(zip(variables, colors()))


def get_avail_data_by_var_heatmap(ds, granularity, adjust_color_intensity_to_max=True, color_mapping=None):
    """

    :param ds: xarray.Dataset or dictionary {var_label: xarray.DataArray};
    :param granularity: str; one of ['year', 'season', 'month']
    :param adjust_color_intensity_to_max: bool, optional, default=True
    :param color_mapping: dict, optional; {var_label: tuple(r, g, b)}, where r, b, g are int's
    :return:
    """
    if color_mapping is None:
        color_mapping = get_color_mapping(ds)

    def get_data_avail_with_freq(ds, granularity):
        if len(list(ds)) == 0:
            return None

        if granularity == 'year':
            freq = 'YS'
        elif granularity == 'season':
            freq = 'QS-DEC'
        elif granularity == 'month':
            freq = 'MS'
        else:
            raise ValueError(f'unknown granularity={granularity}')
        if isinstance(ds, xr.Dataset):
            ds_avail = ds.notnull().resample({'time': freq}).mean()
        else:
            # ds is a dictionary {var_label: xr.DataArray}
            ds_avail = xr.merge(
                [
                    da.reset_coords(drop=True).notnull().resample({'time': freq}).mean().rename(v)
                    for v, da in ds.items()
                ],
                join='outer',
                # compat='override',
            )
        t = ds_avail['time']
        if granularity == 'year':
            t2 = t.dt.year
        elif granularity == 'season':
            season = t.dt.month.to_series().map({12: 'DJF', 3: 'MAM', 6: 'JJA', 9: 'SON'})
            t2 = t.dt.year.astype(str).str.cat('-', xr.DataArray(season))
        elif granularity == 'month':
            month = t.dt.month.to_series().map({m: str(m).zfill(2) for m in range(1, 13)})
            t2 = t.dt.year.astype(str).str.cat('-', xr.DataArray(month))
        else:
            raise ValueError(f'unknown granularity={granularity}')
        ds_avail['time_period'] = t2
        return ds_avail.set_coords('time_period')

    def get_heatmap(ds_avail, adjust_color_intensity_to_max, color_mapping):
        vs = list(reversed(list(ds_avail.data_vars)))
        n_vars = len(vs)
        availability_data = np.stack([ds_avail[v].values for v in vs])

        if not adjust_color_intensity_to_max:
            z_data = availability_data
        else:
            max_availability = np.nanmax(availability_data, axis=1, keepdims=True) if len(availability_data) > 0 else np.nan
            z_data = availability_data / max_availability
            z_data = np.nan_to_num(z_data)
        # this hook is because we want apply different color scale to each row of availability_data:
        z_data += 2 * np.arange(n_vars).reshape((n_vars, 1))
        # and here come the color scales:
        colorscale = [
            # [[2*i / (2*n_vars), f'rgba{color_mapping[v] + (0,)}'], [(2*i+1) / (2*n_vars), f'rgba{color_mapping[v] + (255,)}']]
            [[2*i / (2*n_vars), rgb_to_rgba(color_mapping[v], 0)], [(2*i+1) / (2*n_vars), rgb_to_rgba(color_mapping[v], 1)]]
            for i, v in enumerate(vs)
        ]
        colorscale = sum(colorscale, start=[])
        colorscale.append([1., 'rgba(255, 255, 255, 1)'])  # must define whatever color for z / zmax = 1.
        xperiod0 = None
        if granularity == 'year':
            xperiod = 'M12'
        elif granularity == 'season':
            xperiod = 'M3'
            xperiod0 = ds_avail['time'].values[0]
        elif granularity == 'month':
            xperiod = 'M1'
        else:
            raise ValueError(granularity)
        heatmap = go.Heatmap(
            z=z_data,
            #x=ds_avail['time_period'],
            x=ds_avail['time'].values,
            xperiod=xperiod,
            xperiod0=xperiod0,
            #xperiodalignment='end',
            y=vs,
            colorscale=colorscale,
            customdata=100 * availability_data,   # availability in %
            hovertemplate='%{x}: %{customdata:.0f}%',
            name='',
            showscale=False,
            xgap=1,
            ygap=5,
            zmin=0,
            zmax=2*n_vars,
        )
        return heatmap

    ds_avail = get_data_avail_with_freq(ds, granularity)
    if ds_avail is None:
        return empty_figure()

    n_vars = max(len(ds_avail.data_vars), 1)
    layout_dict = {
        'autosize': True,
        'height': 80 + 30 * n_vars,
        'margin': {'b': 25, 't': 35},
    }

    fig = go.Figure(data=get_heatmap(ds_avail, adjust_color_intensity_to_max, color_mapping), layout=layout_dict)
    if granularity == 'year':
        dtick = 'M12'
        tickformat = '%Y'
    else:
        dtick = 'M3'
        tickformat = '%b %Y'
    fig.update_xaxes(
        type='date',
        dtick=dtick,
        tickformat=tickformat,
        ticklabelmode='period',
        title='time',
    )
    fig.update_layout({'yaxis': {'title': ''}})
    return fig


def get_histogram(da, x_label, bins=50, color=None, x_min=None, x_max=None, log_x=False, log_y=False):
    if da is None:
        return empty_figure(), 1

    color = f'rgb{color}' if isinstance(color, tuple) and len(color) == 3 else color

    ar = da.where(da.notnull(), drop=True).values

    if len(ar) == 0:
        ar_ = np.array([np.nan])
    else:
        ar_ = ar
    qs = np.quantile(ar_, q=[0.25, 0.5, 0.75])
    boxplot_data = {
        'q1': qs[0], 'median': qs[1], 'q3': qs[2],
        'lowerfence': np.amin(ar_), 'upperfence': np.amax(ar_),
        'mean': np.mean(ar_), 'sd': np.std(ar_),
    }
    boxplot_data = {k: [v] for k, v in boxplot_data.items()}
    boxplot_trace = go.Box(
        line={'color': color},
        y=[x_label],
        orientation='h',
        xaxis='x',
        yaxis='y2',
        **boxplot_data
    )

    if log_x:
        ar = np.log(ar[ar > 0])
        x_min = np.log(x_min) if x_min is not None and x_min > 0 else None
        x_max = np.log(x_max) if x_max is not None and x_max > 0 else None

    if np.isnan(x_min):
        x_min = None
    if np.isnan(x_max):
        x_max = None

    rng = [x_min, x_max] if x_min is not None and x_max is not None else None
    h, edges = np.histogram(ar, bins=bins, range=rng)

    if log_x:
        edges = np.exp(edges)

    rng = edges[-1] - edges[0]
    precision = int(np.ceil(np.log10(50 * bins / rng)))
    if precision < 0:
        precision = 0

    histogram_trace = go.Bar(
        name=x_label,
        y=h,
        x=edges[:-1],
        width=np.diff(edges),
        offset=0,
        customdata=np.transpose([edges[:-1], edges[1:], h]),
        hovertemplate='<br>'.join([
            'obs: %{customdata[2]}',
            'range: [%{customdata[0]:.' + str(precision) + 'f}, %{customdata[1]:.' + str(precision) + 'f}]',
        ]),
        marker={'color': color}
    )

    xaxis_title = da.attrs.get('long_name', da.attrs.get('label', '???'))
    xaxis_units = da.attrs.get('units', '???')
    fig_layout = {
        'xaxis': {
            'title': f'{xaxis_title} ({xaxis_units})',
        },
        'yaxis': {
            'title': '# observations',
            'domain': [0, 0.775],
        },
        'yaxis2': {
            'title': '',
            'domain': [0.825, 1],
        }
    }

    fig = go.Figure(data=[histogram_trace, boxplot_trace], layout=fig_layout)

    fig.update_layout({
        'autosize': True,
        'height': 320,
        'margin': {'b': 0, 't': 35},
        'showlegend': False,
    })

    if log_x:
        fig.update_xaxes(type='log')
    if log_y:
        fig.update_layout({'yaxis': {'type': 'log'}})
    return fig, np.max(h) if len(h) > 0 else 1


def align_range(rng, nticks, log_coeffs=(1, 2, 2.5, 5, 10)):
    if nticks < 3:
        ValueError(f'no_ticks must be an integer >= 3; got no_ticks={nticks}')
    nticks = int(nticks)

    log_coeffs = log_coeffs + (10,)
    low, high = rng
    if np.isnan(low) or np.isnan(high) or low >= high:
        return (0, 1), 0, 1 / (nticks - 1)

    dtick = (high - low) / (nticks - 2)
    dtick_base = np.power(10, np.floor(np.log10(dtick)))
    dlog_dtick = dtick / dtick_base
    for log_coeff in log_coeffs:
        if dlog_dtick <= log_coeff:
            break
    dtick = dtick_base * log_coeff
    delta_aligned = dtick * (nticks - 1)
    low_aligned = np.floor(low / dtick) * dtick
    high_aligned = np.ceil(high / dtick) * dtick
    while high_aligned - low_aligned < delta_aligned - dtick / 2:
        high_aligned += dtick
        if high_aligned - low_aligned < delta_aligned - dtick / 2:
            low_aligned -= dtick
    return (low_aligned, high_aligned), low_aligned, dtick


def align_range_containing_zero(rng, nticks_positive, nticks_negative, log_coeffs=(1, 2, 2.5, 5, 10)):
    # print(rng, (nticks_positive, nticks_negative))
    lo, hi = rng
    lo = min(lo, 0.)
    hi = max(hi, 0.)

    assert nticks_positive >= 2 or nticks_positive == 1 and hi == 0, f'invalid nticks_positive={nticks_positive}, with rng={rng}'
    assert nticks_negative >= 2 or nticks_negative == 1 and lo == 0, f'invalid nticks_negative={nticks_negative}, with rng={rng}'

    dtick_positive = hi / (nticks_positive - 1) if hi > 0 else 0
    dtick_negative = -lo / (nticks_negative - 1) if lo < 0 else 0
    dtick = max(dtick_positive, dtick_negative)
    dtick_base = np.power(10, np.floor(np.log10(dtick)))
    # print(dtick, dtick_base)
    dlog_dtick = dtick / dtick_base
    for log_coeff in log_coeffs:
        if dlog_dtick <= log_coeff:
            break
    dtick = dtick_base * log_coeff
    low_aligned, high_aligned = -(nticks_negative - 1) * dtick, (nticks_positive - 1) * dtick
    # print((low_aligned, high_aligned), low_aligned, dtick)
    return (low_aligned, high_aligned), low_aligned, dtick


def get_range_tick0_dtick_by_var(df, nticks, error_df=None, align_zero=False):
    """

    :param df: pandas DataFrame or dict of pandas Series {variable_id: series}, or dict of dict of pandas Series
     {variable_id: {sublabel: series}} (useful e.g. for quantiles);
     in the two last cases each series might have a different index;
     warning: for the moment, cannot use dict of dict of pandas Series together with error_df
    :param nticks:
    :param error_df:
    :return:
    """
    def list_of_series_min_max(list_of_series):
        list_of_series = helper.ensurelist(list_of_series)
        lo = pd.Series([series.min() for series in list_of_series]).min()
        hi = pd.Series([series.max() for series in list_of_series]).max()
        if align_zero:
            lo = min(lo, 0)
            hi = max(hi, 0)
        return [lo, hi]

    range_by_var = toolz.valmap(list_of_series_min_max, df)
    if error_df is not None:
        # adjust min/max so that df + error_df, df - error_df are within min/max
        for v, rng in list(range_by_var.items()):
            if v in error_df:
                lo, hi = rng
                lo_with_std = (df[v] - error_df[v]).min()
                if lo_with_std < lo:
                    lo = lo_with_std
                hi_with_std = (df[v] + error_df[v]).max()
                if hi_with_std > hi:
                    hi = hi_with_std
                if align_zero:
                    lo = min(lo, 0)
                    hi = max(hi, 0)
                range_by_var[v] = [lo, hi]

    if not align_zero:
        range_tick0_dtick_by_var = toolz.valmap(lambda rng: align_range(rng, nticks=nticks), range_by_var)
    else:
        lo, hi = np.array(list(range_by_var.values())).T
        zeros = -lo / (hi - lo)
        zeros = zeros[~np.isnan(zeros)]
        if len(zeros) > 0:
            min_zero = np.min(zeros)
            max_zero = np.max(zeros)
            if min_zero <= .5 and max_zero >= .5:
                zero = .5
            elif max_zero <= .5:
                zero = max_zero
            else:
                zero = min_zero

            new_range_by_var = {}
            _hi, _lo = -1, 1
            for v, (lo, hi) in range_by_var.items():
                if hi > lo:
                    z = -lo / (hi - lo)
                    if z < zero < 1:
                        lo = -hi * zero / (1 - zero)
                    elif 0 < zero < z:
                        hi = -lo * (1 - zero) / zero
                    _hi, _lo = hi, lo
                new_range_by_var[v] = [lo, hi]
            range_by_var = new_range_by_var

            nticks_positive = int(np.ceil((nticks - 1) * _hi / (_hi - _lo))) + 1
            nticks_negative = int(np.ceil((nticks - 1) * (-_lo) / (_hi - _lo))) + 1
            range_tick0_dtick_by_var = toolz.valmap(lambda rng: align_range_containing_zero(rng, nticks_positive, nticks_negative), range_by_var)

    return range_tick0_dtick_by_var


def get_sync_axis_props(range_tick0_dtick_by_var, fixedrange=True):
    axis_props_by_var = {}
    for i, (variable_id, (rng, tick0, dtick)) in enumerate(range_tick0_dtick_by_var.items()):
        axis_props = {
            'range': rng,
            'tick0': tick0,
            'dtick': dtick,
        }
        if i > 0:
            axis_props['overlaying'] = 'y'
        if fixedrange:
            axis_props['fixedrange'] = True
        axis_props_by_var[variable_id] = axis_props
    return axis_props_by_var


# TODO: fix an issue with reseting zoom: seems not always reset to manually defined ranges
def multi_line(
        df,
        df_std=None,
        std_mode='fill',
        width=None,
        height=None,
        scatter_mode='lines',
        nticks=None,
        variable_label_by_var=None,
        yaxis_label_by_var=None,
        color_mapping=None,
        range_tick0_dtick_by_var=None,
        line_dash_style_by_sublabel=None,
        marker_opacity_by_sublabel=None,
        filtering_on_figure_extent=None,
        subsampling=None,
        use_GL='auto',
):
    """

    :param df: pandas DataFrame or dict of pandas Series {variable_id: series}, or dict of dict of pandas Series
     {variable_id: {sublabel: series}} (useful e.g. for quantiles);
     in the two last cases each series might have a different index;
     warning: for the moment, cannot use dict of dict of pandas Series together with df_std
    :param std_mode: 'fill' or 'error_bars' or None
    :param width: int; default 1000
    :param height: int; default 500
    :param scatter_mode: str; 'lines', 'markers' or 'lines+markers'
    :param nticks: int or None; number of ticks on y-axis (or y-axes); default is max(height // 50, 3)
    :param variable_label_by_var: dict of str; labels of variables to be displayed on the plot legend and in hover;
    keys of the dict are that of df (if DataFrame, that's the df columns)
    :param yaxis_label_by_var: dict of str; labels to be displayed as title of y-axis (y-axes);
    keys of the dict are that of df (if DataFrame, that's the df columns)
    :param color_mapping: dict or None;
    :param range_tick0_dtick_by_var: dict or None;
    :param line_dash_style_by_sublabel: None or dict of the form {sublabel: str}, where values are from
    [‘solid’, ‘dot’, ‘dash’, ‘longdash’, ‘dashdot’, ‘longdashdot’]
    :param marker_opacity_by_sublabel: None or dict of the form {sublabel: float}, where floats are in [0, 1]
    :param filtering_on_figure_extent: callable or None; a callable takes a pandas Series as an argument and
    returns either a mask (boolean) series with the same index or True (no masking)
    :param subsampling: int or None; indicates the size of an eventual subsample
    :return:
    """
    df = dict(df)
    nvars = len(list(df))
    if not (nvars >= 1):
        return empty_figure(height=height)

    if nticks is None:
        nticks = max(height // 50, 3) if isinstance(height, (int, float)) else 8
    if variable_label_by_var is None:
        variable_label_by_var = {}
    if yaxis_label_by_var is None:
        yaxis_label_by_var = {}
    if color_mapping is None:
        color_mapping = get_color_mapping(list(df))
    if range_tick0_dtick_by_var is None:
        range_tick0_dtick_by_var = get_range_tick0_dtick_by_var(df, nticks, error_df=df_std)

    fig = go.Figure()

    for i, (variable_id, variable_values_by_sublabel) in enumerate(df.items()):
        variable_label_on_legend = variable_label_by_var.get(variable_id, variable_id)

        if i > 0:
            yaxis = {'yaxis': f'y{i+1}'}
        else:
            yaxis = {}

        if not isinstance(variable_values_by_sublabel, dict):
            variable_values_by_sublabel = {None: variable_values_by_sublabel}
        n_traces = len(variable_values_by_sublabel)

        for j, (sublabel, variable_values) in enumerate(variable_values_by_sublabel.items()):
            if df_std is not None and variable_id in df_std:
                y_std = df_std[variable_id]
                if not y_std.index.equals(variable_values.index):
                    y_std, _ = y_std.align(variable_values, join='right')
            else:
                y_std = None

            extra_kwargs = {}
            if n_traces == 1:
                extra_kwargs['name'] = variable_label_on_legend
                # if sublabel is not None:
                #     extra_kwargs['name'] = extra_kwargs['name'] + f' ({sublabel})'
            else:
                if j == 0:
                    extra_kwargs['legendgrouptitle_text'] = variable_label_on_legend
                extra_kwargs['name'] = sublabel
            if n_traces > 1 and 'lines' in scatter_mode and line_dash_style_by_sublabel is not None:
                line_dash_style = line_dash_style_by_sublabel.get(sublabel)
                if line_dash_style is None:
                    line_dash_style = line_dash_style_by_sublabel.get('other')
                if line_dash_style is not None:
                    extra_kwargs['line_dash'] = line_dash_style

            color = plotly.colors.label_rgb(color_mapping[variable_id])

            if n_traces > 1 and marker_opacity_by_sublabel is not None:
                color_opacity = marker_opacity_by_sublabel.get(sublabel)
                if color_opacity is not None:
                    color = rgb_to_rgba(color, color_opacity)

            if filtering_on_figure_extent is not None:
                figure_extent_mask = filtering_on_figure_extent(variable_values)
                if figure_extent_mask is not True:
                    variable_values = variable_values[figure_extent_mask]
                    if y_std is not None:
                        y_std = y_std[figure_extent_mask]

            if subsampling is not None:
                subsampling_mask = get_subsampling_mask(~np.isnan(variable_values.values), n=subsampling)
                variable_values = variable_values[subsampling_mask]
                if y_std is not None:
                    y_std = y_std[subsampling_mask]

            scatter = plotly_scatter(
                x=variable_values.index.values,
                y=variable_values.values,
                y_std=y_std.values if y_std is not None else None,
                std_mode=std_mode,
                legendgroup=variable_id,
                mode=scatter_mode,
                marker_color=color,
                use_GL=use_GL,
                **yaxis,
                **extra_kwargs,
            )
            if isinstance(scatter, tuple):
                fig.add_traces(scatter)
            else:
                fig.add_trace(scatter)

    delta_domain = min(75 / width, 0.5 / nvars) if isinstance(width, (int, float)) else 0.25 / nvars
    domain = [delta_domain * ((nvars - 1) // 2), min(1 - delta_domain * ((nvars - 2) // 2), 1)]
    fig.update_layout(xaxis={'domain': domain})

    yaxis_props_by_var = get_sync_axis_props(range_tick0_dtick_by_var, fixedrange=False)

    for i, (variable_id, yaxis_props) in enumerate(yaxis_props_by_var.items()):
        yaxis_props.update({
            #'rangeslider': {'visible': True},
            #'gridcolor': 'black',
            #'gridwidth': 1,
            'tickcolor': f'rgb{color_mapping[variable_id]}',
            'ticklabelposition': 'outside',
            'tickfont_color': f'rgb{color_mapping[variable_id]}',
            #'minor_showgrid': False,
            'title': {
                'font_color': f'rgb{color_mapping[variable_id]}',
                'standoff': 0,
                'text': yaxis_label_by_var.get(variable_id),
            },
            'showline': True,
            'linewidth': 2,
            'linecolor': f'rgb{color_mapping[variable_id]}',
            'zeroline': True,
            'zerolinewidth': 1,
            #'zerolinecolor': 'black',
            # 'automargin': True, # ???
        })

        idx_of_last_variable_on_left_side = (nvars + 1) // 2 - 1
        idx_of_first_variable_on_right_side = (nvars + 1) // 2
        if i == idx_of_last_variable_on_left_side:
            yaxis_props.update({
                'side': 'left',
                #'ticks': 'inside',
            })
        elif i == idx_of_first_variable_on_right_side:
            yaxis_props.update({
                'anchor': 'x',
                'side': 'right',
                #'ticks': 'inside',
            })
        else:
            if i < idx_of_last_variable_on_left_side:
                position = delta_domain * i
                side = 'left'
            else:
                position = 1 - delta_domain * (nvars - 1 - i)
                side = 'right'
            yaxis_props.update({
                'anchor': 'free',
                'position': position,
                'side': side,
            })

        yaxis_id = 'yaxis' if i == 0 else f'yaxis{i+1}'
        fig.update_layout({yaxis_id: yaxis_props})

    # fig_size = {'minreducedwidth': 500}
    fig_size = {}
    if height:
        #fig_size['minreducedheight'] = height
        fig_size['height'] = height
    if isinstance(width, (int, float)):
        #fig_size['minreducedwidth'] = width
        fig_size['width'] = width
    if fig_size:
        fig.update_layout(fig_size)

    # customize legend
    fig.update_layout(legend={
        'orientation': 'h',
        'xanchor': 'left',
        'yanchor': 'top',
        'x': 0,
        'y': -0.2,
    })

    return fig


def plotly_scatter2d(
        x, y, C=None,
        cmin=None, cmax=None,
        cmap='jet',
        xaxis_title=None,
        yaxis_title=None,
        colorbar_title=None,
        height=400, width=520,
        margin=dict(b=40, t=30, l=20, r=20),
        marker_size=5,
):
    if C is not None:
        colorbar = {
            'ticklen': 6,
            'ticks': 'inside',
            'thickness': 20,
            # 'orientation': 'h',
            # 'x': 0.5, 'y': -0.2,
            'title': {
                'text': colorbar_title,
                'side': 'right'
            },
        }
        marker_kwargs = {
            'color': C,
            'colorscale': cmap,
            'showscale': True,
            'colorbar': colorbar,
        }
        if cmin is not None and cmax is not None:
            marker_kwargs.update({'cmin': cmin, 'cmax': cmax})
    else:
        marker_kwargs = {}
    marker_kwargs.update({'size': marker_size})

    trace = go.Scattergl(
        x=x, y=y,
        mode='markers',
        marker={
            #'size': 0.01,
            **marker_kwargs
        },
    )
    fig = go.Figure(data=trace)

    axis = dict(
        showgrid=True,
        showline=True,
        # zeroline=False,
        ticks='inside',
        ticklen=6,
    )

    if xaxis_title is not None:
        xaxis = axis.copy()
        xaxis.update(title=xaxis_title)
    else:
        xaxis = axis
    if yaxis_title is not None:
        yaxis = axis.copy()
        yaxis.update(title=yaxis_title)
    else:
        yaxis = axis

    fig.update_layout({
        'width': width,
        'height': height,
        'xaxis': xaxis,
        'yaxis': yaxis,
        # 'hovermode': 'closest',
        'margin': margin,
    })

    return fig


def _get_hexagonal_binning(
        x, y,
        C=None,
        reduce_C_function=None,
        gridsize=20,
        aspect_ratio=1,
):
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    mask_notnan = ~(np.isnan(x) | np.isnan(y))
    if C is not None:
        mask_notnan &= ~np.isnan(C)
    x = x[mask_notnan]
    y = y[mask_notnan]
    if C is not None:
        C = C[mask_notnan]

    if x.shape[0] == 0:
        raise EmptyFigureException()

    p_min = np.array([np.amin(x), np.amin(y)])
    p_max = np.array([np.amax(x), np.amax(y)])
    p0 = (p_max + p_min) / 2

    if isinstance(gridsize, (list, tuple)):
        x_gridsize, y_gridsize = gridsize
        y_gridsize *= 2
    else:
        x_gridsize = gridsize
        y_gridsize = int(x_gridsize / aspect_ratio / np.sqrt(3) * 2)

    if x_gridsize % 2 == 0:
        x_gridsize -= 1
    if y_gridsize % 2 == 0:
        y_gridsize -= 1

    resol = np.array([x_gridsize, y_gridsize])
    scale = np.maximum((p_max - p_min), 1) / resol

    e = np.array([[1, 0], [-0.5, np.sqrt(3) / 2]])
    e_scaled = e * scale
    e_scaled_inv = np.linalg.inv(e_scaled).T

    i = (
            e_scaled_inv[:, 0, np.newaxis] * x + e_scaled_inv[:, 1, np.newaxis] * y
            - (e_scaled_inv[:, :] * p0).sum(axis=1)[:, np.newaxis]
    ).round().astype('i4')

    df_dict = {'i': i[0, :], 'j': i[1, :]}
    if C is not None:
        df_dict['C'] = C
    else:
        df_dict['C'] = 0  # just to apply count on the column 'C'
    df = pd.DataFrame.from_dict(df_dict)
    df_grouped = df['C'].groupby(by=[df['i'], df['j']])

    df_agg = {}
    if reduce_C_function is not None:
        for label, f in reduce_C_function.items():
            df_agg[label] = f(df_grouped)
    else:
        df_agg['count'] = df_grouped.count()
    df_agg = pd.DataFrame.from_dict(df_agg)

    i = df_agg.index.get_level_values('i')
    j = df_agg.index.get_level_values('j')
    ij = np.vstack((i.values, j.values))
    centers = p0[:, np.newaxis] + (ij[:, np.newaxis, :] * e_scaled[:, :, np.newaxis]).sum(axis=0)

    _hexagon_x_coords = np.array([1, 0, -1, -1, 0, 1]) * 0.5
    _hexagon_y_coords = np.array([1, 2, 1, -1, -2, -1]) * np.sqrt(3) / 6
    hexagon = np.vstack((_hexagon_x_coords, _hexagon_y_coords))

    return centers.T, df_agg, (hexagon * scale[:, np.newaxis]).T


# @log_exectime
def plotly_hexbin(
        x, y, C=None,
        mode=None,  # ['2d', '3d', '3d+sample_size', '3d+sample_size_as_hexagon_scaling']
        reduce_function=np.mean,
        cmin=None, cmax=None,
        min_count=1,
        gridsize=50,
        cmap='jet',
        sample_size_transform=np.log1p,
        sample_size_inverse_transform=np.expm1,
        opaque_hexagon_centers=0.3,      # only used when C is not None
        xaxis_title=None,
        yaxis_title=None,
        colorbar_title=None,
        height=400, width=520,
        margin=(('b', 40), ('t', 30), ('l', 20), ('r', 20))
):
    # TODO: manages missing values in x, y and C (do sth like ds.dropna? ask user to ensure this?)
    # TODO: after filtering with min_count, the plot bbox can change (it particular its aspect ratio) and in consequence, aspect ratio of hexagon changes? can remedy this?

    margin = dict(margin)
    plot_width = max(width - 20 - margin['l'] - margin['r'], 10)
    plot_height = max(height - margin['t'] - margin['b'], 10)

    if mode is None:
        mode = '2d' if C is None else '3d'

    mode_options = ['2d', '3d', '3d+sample_size', '3d+sample_size_as_hexagon_scaling']
    assert mode in mode_options, \
        f'mode must be one of {mode_options}; got mode={mode}'
    assert mode == '2d' or C is not None, \
        f'C is None, so mode must be "2d"; got mode={mode}'
    assert mode == '2d' or reduce_function is not None, \
        f'mode={mode}, so reduce_function cannot be None'
    assert mode == '3d' or sample_size_transform is not None, \
        f'mode={mode}, so sample_size_transform cannot be None'
    assert mode != '2d' or sample_size_inverse_transform is not None, \
        f'mode is "2d", so sample_size_inverse_transform cannot be None'

    if mode != '2d':
        reduce_C_function = {
            'C': reduce_function,
            'q5': lambda a: pd.core.groupby.DataFrameGroupBy.quantile(a, q=0.1),
            # 'q5': lambda a: pd.core.groupby.DataFrameGroupBy.min(a),
            # 'q95': lambda a: pd.core.groupby.DataFrameGroupBy.max(a),
            'q95': lambda a: pd.core.groupby.DataFrameGroupBy.quantile(a, q=0.9),
            'count': lambda a: pd.core.groupby.DataFrameGroupBy.count(a)
        }
    else:
        reduce_C_function = None

    centers, values, hexagon_vertices = _get_hexagonal_binning(
        x, y,
        C=C if mode != '2d' else None,
        reduce_C_function=reduce_C_function,
        gridsize=gridsize,
        aspect_ratio=plot_width / plot_height,
    )

    counts = values['count'].values
    _hexagons = centers[:, np.newaxis, :] + hexagon_vertices[np.newaxis, :, :]  # _hexagon, vertex, 2d-coord: shape is (n, 6, 2)

    if mode in ['2d', '3d+sample_size', '3d+sample_size_as_hexagon_scaling']:
        max_count = np.amax(counts) if len(counts) > 0 else 1
        sample_size_transformed = sample_size_transform(counts)
        max_sample_size_transformed = sample_size_transform(np.array(max_count))
        relative_sample_size = sample_size_transformed / max_sample_size_transformed

    if mode in ['2d', '3d', '3d+sample_size']:
        hexagons = _hexagons
        if mode == '3d+sample_size':
            if opaque_hexagon_centers is not None and opaque_hexagon_centers > 0:
                opaque_hexagons = opaque_hexagon_centers * _hexagons \
                                  + (1 - opaque_hexagon_centers) * np.expand_dims(centers, 1)
            else:
                opaque_hexagons = None
    elif mode == '3d+sample_size_as_hexagon_scaling':
        _relative_sample_size = np.expand_dims(relative_sample_size, (1, 2))
        hexagons = _relative_sample_size * _hexagons \
                   + (1 - _relative_sample_size) * np.expand_dims(centers, 1)
    else:
        raise ValueError(f'mode={mode}')

    if mode != '2d':
        z = values['C'].values
        z_high = cmax
        if z_high is None:
            q95 = values['q95'].values
            z_high = np.amax(q95) if len(q95) > 0 else 1
        z_low = cmin
        if z_low is None:
            q5 = values['q5'].values
            z_low = np.amin(q5) if len(q5) > 0 else 0
    else:  # mode == '2d'
        z = sample_size_transformed
        z_high = max_sample_size_transformed
        z_low = sample_size_transform(np.array(0))

    z_relative = (z - z_low) / (z_high - z_low)
    # in case z_high == z_low:
    z_relative = np.nan_to_num(z_relative, nan=0.5, posinf=0.5, neginf=0.5)
    colors = plotly.colors.sample_colorscale(cmap, np.clip(z_relative, 0, 1), colortype='tuple')
    colors = np.round(np.array(colors) * 255).astype('i4')

    if mode == '3d+sample_size':
        hexagon_colors = [f'rgba({r}, {g}, {b}, {a})' for (r, g, b), a in zip(colors, relative_sample_size)]
        if opaque_hexagons is not None:
            opaque_hexagon_colors = [f'rgb({r}, {g}, {b})' for r, g, b in colors]
    else:
        hexagon_colors = [f'rgb({r}, {g}, {b})' for r, g, b in colors]

    hexagon_x_centers, hexagon_y_centers = centers[:, 0], centers[:, 1]
    if mode != '2d':
        hover_text = [
            f'x: {hexagon_x_center:.4g}<br>'
            f'y: {hexagon_y_center:.4g}<br>'
            f'c: {c:.4g}<br>'
            f'samples: {int(count):.4g}'
            for hexagon_x_center, hexagon_y_center, c, count in zip(hexagon_x_centers, hexagon_y_centers, z, counts)
        ]
    else:  # mode == '2d'
        hover_text = [
            f'x: {hexagon_x_center:.4g}<br>'
            f'y: {hexagon_y_center:.4g}<br>'
            f'samples: {int(count):.4g}'
            for hexagon_x_center, hexagon_y_center, count in zip(hexagon_x_centers, hexagon_y_centers, counts)
        ]

    colorbar = {
        'ticklen': 6,
        'ticks': 'inside',
        'thickness': 20,
        # 'orientation': 'h',
        # 'x': 0.5, 'y': -0.2,
        'title': {
            'text': colorbar_title,
            'side': 'right'
        },
    }
    if mode == '2d':
        tickvals = np.linspace(z_low, z_high, 11)
        ticktext = [f'{sample_size_inverse_transform(_z):.4g}' for _z in tickvals]
        colorbar.update({
            'tickvals': tickvals,
            'ticktext': ticktext,
        })

    trace = go.Scatter(
        x=hexagon_x_centers,
        y=hexagon_y_centers,
        mode='markers',
        marker={
            'size': 0.01,
            'color': z,
            'cmin': z_low,
            'cmax': z_high,
            'colorscale': cmap,
            'showscale': True,
            'colorbar': colorbar,
        },
        text=hover_text,
        hoverinfo='text',
    )

    def get_shape_path(hexagon, color):
        path = 'M '
        path += ' L '.join(f'{v[0]}, {v[1]}' for v in hexagon)
        path += ' Z'
        shape = {
            'type': 'path',
            'path': path,
            'fillcolor': color,
            'line_width': 0,
        }
        return shape

    shapes = []
    for hexagon, hexagon_color in zip(hexagons, hexagon_colors):
        shapes.append(get_shape_path(hexagon, hexagon_color))

    if mode == '3d+sample_size' and opaque_hexagons is not None:
        for opaque_hexagon, color_rgb in zip(opaque_hexagons, opaque_hexagon_colors):
            shapes.append(get_shape_path(opaque_hexagon, color_rgb))

    # x_min, x_max = (x.min(), x.max()) if len(x) > 0 else (0, 1)
    # y_min, y_max = (y.min(), y.max()) if len(y) > 0 else (0, 1)

    fig = go.Figure(data=trace)
    # fig.update_xaxes({
    #    'range': [x_min, x_max],
    # })
    # fig.update_yaxes({
    #    'range': [y_min, y_max],
    # })
    axis = dict(
        showgrid=False,
        showline=False,
        zeroline=False,
        ticks='inside',
        ticklen=6,
    )

    if xaxis_title is not None:
        xaxis = axis.copy()
        xaxis.update(title=xaxis_title)
    else:
        xaxis = axis
    if yaxis_title is not None:
        yaxis = axis.copy()
        yaxis.update(title=yaxis_title)
    else:
        yaxis = axis

    fig.update_layout({
        'width': width,
        'height': height,
        'xaxis': xaxis,
        'yaxis': yaxis,
        'hovermode': 'closest',
        'shapes': shapes,
        'margin': margin,
    })

    return fig


def empty_figure(height=None):
    fig = go.Figure()
    if height is not None:
        fig.update_layout(height=height)
    return fig


def _get_watermark_size(fig):
    if not isinstance(fig, dict):
        fig = fig.to_dict()

    default_size = 75
    ref_height = 500
    ref_width = 1000

    layout = fig.get('layout')
    if layout is None:
        return default_size
    height = layout.get('height')
    if height is not None:
        return default_size * height / ref_height
    width = layout.get('width', ref_width)
    return default_size * width / ref_width


def _get_fig_center(fig):
    if not isinstance(fig, dict):
        fig = fig.to_dict()

    default_center_by_axis = {
        'xaxis': .5,
        'yaxis': .5,
    }
    def_center = (default_center_by_axis['xaxis'], default_center_by_axis['yaxis'])

    layout = fig.get('layout')
    if layout is None:
        return def_center

    def get_axis_domain_center(axis):
        _axis = layout.get(axis)
        if _axis is None:
            return default_center_by_axis[axis]
        return sum(_axis.get('domain', (0, 1))) / 2

    x = get_axis_domain_center('xaxis')
    y = get_axis_domain_center('yaxis')
    return x, y


def add_watermark(fig, size=None):
    if size is None:
        size = _get_watermark_size(fig)
    #x, y = _get_fig_center(fig)
    x, y = 0.5, 0.5

    annotations = [dict(
        name="watermark",
        text="ATMO-ACCESS",
        textangle=-30,
        opacity=0.05,
        font=dict(color="black", size=size),
        xref="paper",
        yref="paper",
        #xref='x domain',
        #yref='y domain',
        x=x,
        y=y,
        showarrow=False,
    )]
    fig.update_layout(annotations=annotations)
    return fig


def apply_figure_xaxis_extent(fig, relayout_data):
    if relayout_data is not None:
        rng = [relayout_data.get(f'xaxis.range[{i}]') for i in range(2)]
        if all(r is not None for r in rng):
            fig.update_layout(xaxis={'range': rng})
    return fig


def get_figure_extent(relayout_data):
    """

    :param relayout_data:
    :return: dict or True; True if autosize=True is within relayout_data
    """
    print(f'get_figure_extent(relayout_data={relayout_data})')

    if relayout_data is not None:
        layout_dict = {}
        try:
            for k, v in relayout_data.items():
                try:
                    if k in ['dragmode', 'selections']:
                        continue
                    if k == 'autosize':
                        if v:
                            return True
                        else:
                            continue

                    k_split = k.split('.')
                    if len(k_split) != 2:
                        continue
                    axis, rng = k_split

                    if rng.startswith('range[') and len(rng) == 8 and rng[-1] == ']':
                        i = int(rng[-2])
                        assert i in [0, 1]
                        layout_dict.setdefault(axis, {'range': [None, None]})['range'][i] = v
                    elif rng == 'autorange' and v:
                            layout_dict[axis] = {'range': [None, None]}
                    else:
                        continue

                except Exception as e:
                    logger().exception(f'Failed to parse relayout_data item k={k}, v={v}; relayout_data={relayout_data}', exc_info=e)
            return layout_dict
        except Exception as e:
            logger().exception(f'Failed to parse relayout_data={relayout_data}', exc_info=e)
    return None


def apply_figure_extent(fig, relayout_data):
    """

    :param fig: plotly.graphical_objects.Figure
    :param relayout_data: dict, e.g. relayout_data={'xaxis.range[0]': '1987-03-28 07:16:06.9794', 'xaxis.range[1]': '2003-01-18 00:53:57.7486'}
    :return: plotly.graphical_objects.Figure
    """
    layout_dict = get_figure_extent(relayout_data)
    # print(f'charts.apply_figure_extent: relayout_data={relayout_data}, layout_dict={layout_dict}')
    if isinstance(layout_dict, dict) and layout_dict:
        fig.update_layout(layout_dict)
    return fig


def filter_ds_on_xy_extent(ds, figure_extent, x_var=None, y_var=None, x_rel_margin=None, y_rel_margin=None):
    xy_extent_cond = True
    xy_extent_cond_as_str = None

    if isinstance(figure_extent, dict):
        # apply x- and y-data filtering according to figure extent (zoom)
        if x_var is not None:
            x_min, x_max = figure_extent.get('xaxis', {}).get('range', [None, None])
        else:
            x_min, x_max = None, None
        x_margin = (x_max - x_min) * x_rel_margin if not helper.any_is_None(x_min, x_max, x_rel_margin) else 0

        if y_var is not None:
            y_min, y_max = figure_extent.get('yaxis', {}).get('range', [None, None])
        else:
            y_min, y_max = None, None
        y_margin = (y_max - y_min) * y_rel_margin if not helper.any_is_None(y_min, y_max, y_rel_margin) else 0

        xy_extent_cond_as_str = []

        if x_min is not None:
            xy_extent_cond &= (ds[x_var] >= x_min - x_margin)
        if x_max is not None:
            xy_extent_cond &= (ds[x_var] <= x_max + x_margin)

        if x_min is not None and x_max is not None:
            xy_extent_cond_as_str.append(f'{x_min:.4g} <= X <= {x_max:.4g}')
        elif x_min is not None:
            xy_extent_cond_as_str.append(f'{x_min:.4g} <= X')
        elif x_max is not None:
            xy_extent_cond_as_str.append(f'X <= {x_max:.4g}')

        if y_min is not None:
            xy_extent_cond &= (ds[y_var] >= y_min - y_margin)
        if y_max is not None:
            xy_extent_cond &= (ds[y_var] <= y_max + y_margin)

        if y_min is not None and y_max is not None:
            xy_extent_cond_as_str.append(f'{y_min:.4g} <= Y <= {y_max:.4g}')
        elif y_min is not None:
            xy_extent_cond_as_str.append(f'{y_min:.4g} <= Y')
        elif y_max is not None:
            xy_extent_cond_as_str.append(f'Y <= {y_max:.4g}')

        if xy_extent_cond is not True:
            xy_extent_cond_as_str = ' and '.join(xy_extent_cond_as_str)

    return xy_extent_cond, xy_extent_cond_as_str


def filter_series_on_x_extent(series, figure_extent, time_margin=None):
    if time_margin is None:
        time_margin = np.timedelta64(0)

    cond = True
    if isinstance(figure_extent, dict):
        # apply x- and y-data filtering according to figure extent (zoom)
        x_min, x_max = figure_extent.get('xaxis', {}).get('range', [None, None])
        time = series.index.to_series()

        if x_min is not None:
            cond &= (time >= pd.Timestamp(x_min).to_datetime64() - time_margin)
        if x_max is not None:
            cond &= (time <= pd.Timestamp(x_max).to_datetime64() + time_margin)

    return cond
