import toolz
import numpy as np
import pandas as pd
import xarray as xr
from plotly import express as px, graph_objects as go
import plotly.colors

from log.log import logger


# Color codes
ACTRIS_COLOR_HEX = '#00adb7'
IAGOS_COLOR_HEX = '#456096'
ICOS_COLOR_HEX = '#ec165c'


def rgb_to_rgba(rgb, opacity):
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
    rgba_tuple = rgb_tuple + (opacity, )
    return f'rgba{rgba_tuple}'


def plotly_scatter(x, y, *args, y_std=None, std_mode=None, std_fill_opacity=0.2, **kwargs):
    """
    This is a thin wrapper around plotly.graph_objects.Scatter. It workarounds plotly bug:
    Artifacts on line scatter plot when the first item is None #3959
    https://github.com/plotly/plotly.py/issues/3959

    :param std_mode: 'fill', 'error_bars' or None;
    """
    assert std_mode in ['fill', 'error_bars', None]

    x = np.asanyarray(x)
    y = np.asanyarray(y)
    y_isnan = np.isnan(y).astype('i4')
    isolated_points = np.diff(y_isnan, n=2, prepend=1, append=1) == 2
    y_without_isolated_points = np.where(~isolated_points, y, np.nan)

    # print(f'plotly_scatter: len(x)={len(x)}')

    # TODO: test if we can mix traces of scatter and scattergl type on a single go.Figure
    if len(x) <= 1_000:
        go_Scatter = go.Scatter
        go_scatter_ErrorY = go.scatter.ErrorY
    else:
        # for many points, we use GL version: https://github.com/plotly/plotly.js/issues/741
        # but there is a bug with fill='tonexty': https://github.com/plotly/plotly.py/issues/2018,
        #   https://github.com/plotly/plotly.py/issues/2322, https://github.com/plotly/plotly.js/issues/4017
        # for further improvements see: https://github.com/plotly/plotly.js/issues/6230 (e.g. layout.hovermode = 'x')
        go_Scatter = go.Scattergl
        go_scatter_ErrorY = go.scattergl.ErrorY

    if y_std is not None and std_mode == 'error_bars':
        kwargs = kwargs.copy()
        y_std = np.asanyarray(y_std)
        y_std_without_isolated_points_y = np.where(~isolated_points, y_std, np.nan)
        kwargs['error_y'] = go_scatter_ErrorY(array=y_std_without_isolated_points_y, symmetric=True, type='data')

    y_scatter = go_Scatter(
        x=x,
        y=y_without_isolated_points,
        *args,
        **kwargs
    )
    if y_std is None or std_mode != 'fill':
        return y_scatter
    else:
        # y_std is not None ad std_mode == 'fill'
        y_std = np.asanyarray(y_std)

        kwargs_lo = kwargs.copy()
        kwargs_lo['mode'] = 'lines'
        kwargs_lo.pop('line_width', None)
        kwargs_lo.setdefault('line', {})
        kwargs_lo['line']['width'] = 0
        kwargs_lo['hoverinfo'] = 'skip'
        kwargs_lo['showlegend'] = False

        y_max = np.nanmax(np.abs(y) + y_std)
        # this is a not very elegant workaround to the plotly bug https://github.com/plotly/plotly.js/issues/2736
        y_lo = np.nan_to_num(y - y_std, nan=y_max * 1e10)
        y_scatter_lo = plotly_scatter(x, y_lo, *args, **kwargs_lo)

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
        # this is a not very elegant workaround to the plotly bug https://github.com/plotly/plotly.js/issues/2736
        y_hi = np.nan_to_num(y + y_std, nan=y_max * 1e10)
        y_scatter_hi = plotly_scatter(x, y_hi, *args, **kwargs_hi)

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
    gantt.update_layout(
        clickmode='event+select',
        selectdirection='h',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.04, 'xanchor': 'left', 'x': 0},
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
    gantt = px.timeline(
        df, x_start='time_period_start', x_end='time_period_end', y='platform_id_RI', color='var_codes_filtered',
        hover_name='station_fullname',
        hover_data={'station_fullname': True, 'platform_id_RI': True, 'var_codes_filtered': True, 'datasets': True},
        custom_data=['indices'],
        height=height, facet_col='var_codes_filtered', facet_col_wrap=facet_col_wrap,
    )
    gantt.update_layout(
        clickmode='event+select',
        selectdirection='h',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.06, 'xanchor': 'left', 'x': 0},
    )
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
        category_orders={'RI': ['ACTRIS', 'IAGOS', 'ICOS']},
        color_discrete_sequence=[ACTRIS_COLOR_HEX, IAGOS_COLOR_HEX, ICOS_COLOR_HEX],
        height=height
    )
    gantt.update_layout(
        clickmode='event',
        selectdirection='h',
        legend={'orientation': 'h', 'yanchor': 'bottom', 'y': 1.04, 'xanchor': 'left', 'x': 0},
    )
    return gantt


def colors():
    list_of_rgb_triples = [plotly.colors.hex_to_rgb(hex_color) for hex_color in px.colors.qualitative.Dark24]
    return list_of_rgb_triples


def get_color_mapping(variables):
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
    return fig


def get_histogram(da, x_label, bins=50, color=None, x_min=None, x_max=None, log_x=False, log_y=False):
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
    return fig


def align_range(rng, nticks, log_coeffs=(2, 2.5, 5)):
    if nticks < 3:
        ValueError(f'no_ticks must be an integer >= 3; got no_ticks={nticks}')
    nticks = int(nticks)

    log_coeffs = log_coeffs + (10,)
    low, high = rng
    if np.isnan(low) or np.isnan(high):
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


def multi_line(
        df,
        df_std=None,
        std_mode='fill',
        width=1000, height=500,
        scatter_mode='lines',
        nticks=None,
        variable_label_by_var=None,
        yaxis_label_by_var=None,
        color_mapping=None,
        range_tick0_dtick_by_var=None,
        line_dash_style_by_sublabel=None,
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
    :return:
    """
    def ensurelist(obj):
        if isinstance(obj, list):
            return obj
        elif isinstance(obj, tuple):
            return list(obj)
        elif isinstance(obj, dict):
            return list(obj.values())
        else:
            return [obj]

    df = dict(df)
    nvars = len(list(df))
    if not (nvars >= 1):
        return None
    if nticks is None:
        nticks = max(height // 50, 3)
    if variable_label_by_var is None:
        variable_label_by_var = {}
    if yaxis_label_by_var is None:
        yaxis_label_by_var = {}
    if color_mapping is None:
        color_mapping = get_color_mapping(list(df))
    if range_tick0_dtick_by_var is None:
        def list_of_series_min_max(list_of_series):
            list_of_series = ensurelist(list_of_series)
            lo = pd.Series([series.min() for series in list_of_series]).min()
            hi = pd.Series([series.max() for series in list_of_series]).max()
            return lo, hi

        range_by_var = toolz.valmap(list_of_series_min_max, df)
        if df_std is not None:
            # adjust min/max of y-axis so that mean+std, mean-std are within min/max
            for v, rng in list(range_by_var.items()):
                if v in df_std:
                    lo, hi = rng
                    lo_with_std = (df[v] - df_std[v]).min()
                    if lo_with_std < lo:
                        lo = lo_with_std
                    hi_with_std = (df[v] + df_std[v]).max()
                    if hi_with_std > hi:
                        hi = hi_with_std
                    range_by_var[v] = (lo, hi)

        range_tick0_dtick_by_var = toolz.valmap(lambda rng: align_range(rng, nticks=nticks), range_by_var)

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
                assert y_std.index.equals(variable_values.index)
                y_std = y_std.values
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

            scatter = plotly_scatter(
                x=variable_values.index.values,
                y=variable_values.values,
                y_std=y_std,
                std_mode=std_mode,
                legendgroup=variable_id,
                mode=scatter_mode,
                marker_color=plotly.colors.label_rgb(color_mapping[variable_id]),
                **yaxis,
                **extra_kwargs,
            )
            if isinstance(scatter, tuple):
                fig.add_traces(scatter)
            else:
                fig.add_trace(scatter)

    delta_domain = min(75 / width, 0.5 / nvars)
    domain = [delta_domain * ((nvars - 1) // 2), min(1 - delta_domain * ((nvars - 2) // 2), 1)]
    fig.update_layout(xaxis={'domain': domain})

    for i, (variable_id, (rng, tick0, dtick)) in enumerate(range_tick0_dtick_by_var.items()):
        yaxis_props = {
            #'gridcolor': 'black',
            #'gridwidth': 1,
            'range': rng,
            'tick0': tick0,
            'dtick': dtick,
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
            'fixedrange': True,
        }
        if i > 0:
            yaxis_props.update({'overlaying': 'y'})

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

    fig_size = {}
    if height:
        #fig_size['minreducedheight'] = height
        fig_size['height'] = height
    if width:
        #fig_size['minreducedwidth'] = width
        fig_size['width'] = width
    if fig_size:
        fig.update_layout(fig_size)

    return fig


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
        margin=dict(b=40, t=30, l=20, r=20),
):
    # TODO: manages missing values in x, y and C (do sth like ds.dropna? ask user to ensure this?)
    # TODO: after filtering with min_count, the plot bbox can change (it particular its aspect ratio) and in consequence, aspect ratio of hexagon changes? can remedy this?

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

    import matplotlib.pyplot as plt
    # dx = x.max() - x.min()
    # dy = y.max() - y.min()

    if isinstance(gridsize, int):
        # define gridsize(x, y) so that the hexagons are approximately regular
        # see: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hexbin.html
        plot_width = max(width - 20 - margin['l'] - margin['r'], 10)
        plot_height = max(height - margin['t'] - margin['b'], 10)
        gridsize_x = gridsize
        gridsize_y = max(round(gridsize_x / np.sqrt(3) * plot_height / plot_width), 1)
        gridsize = (gridsize_x, gridsize_y)

    mpl_hexbin = plt.hexbin(x, y, gridsize=gridsize, mincnt=max(min_count, 1))
    #plt.close()
    counts = np.asarray(mpl_hexbin.get_array())
    offsets = mpl_hexbin.get_offsets()
    _hexagon, = mpl_hexbin.get_paths()
    hexagon_vertices = np.array([vertex for vertex, _ in _hexagon.iter_segments()][:-1])

    if mode != '2d':
        mpl_hexbin = plt.hexbin(
            x, y, C=C,
            reduce_C_function=reduce_function,
            gridsize=gridsize,
            mincnt=max(min_count - 1, 0)
        )
        #plt.close()
        cs = np.asarray(mpl_hexbin.get_array())
        _offsets = mpl_hexbin.get_offsets()
        _hexagon, = mpl_hexbin.get_paths()
        _hexagon_vertices = np.array([vertex for vertex, _ in _hexagon.iter_segments()][:-1])
        np.testing.assert_allclose(_offsets, offsets)
        np.testing.assert_allclose(_hexagon_vertices, hexagon_vertices)

    _hexagons = np.expand_dims(offsets, 1) + np.expand_dims(hexagon_vertices, 0)  # _hexagon, vertex, 2d-coord: shape is (n, 6, 2)
    centers = _hexagons.mean(axis=1)

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
        z = cs
        z_high = cmax
        if z_high is None:
            z_high = np.amax(z) if len(z) > 0 else 1
        z_low = cmin
        if z_low is None:
            z_low = np.amin(z) if len(z) > 0 else 0
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
            for hexagon_x_center, hexagon_y_center, c, count in zip(hexagon_x_centers, hexagon_y_centers, cs, counts)
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


def empty_figure():
    return go.Figure()


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
        axis = layout.get(axis)
        if axis is None:
            return default_center_by_axis[axis]
        return sum(axis.get('domain', (0, 1))) / 2

    x = get_axis_domain_center('xaxis')
    y = get_axis_domain_center('yaxis')
    return x, y


def add_watermark(fig, size=None):
    if size is None:
        size = _get_watermark_size(fig)
    x, y = _get_fig_center(fig)

    annotations = [dict(
        name="watermark",
        text="ATMO-ACCESS",
        textangle=-30,
        opacity=0.05,
        font=dict(color="black", size=size),
        xref="paper",
        yref="paper",
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
                            raise RuntimeError('unknown relayout command 1')

                    axis, rng = k.split('.')

                    if rng.startswith('range[') and len(rng) == 8 and rng[-1] == ']':
                        i = int(rng[-2])
                        assert i in [0, 1]
                        layout_dict.setdefault(axis, {'range': [None, None]})['range'][i] = v
                    elif rng == 'autorange':
                        if v:
                            layout_dict[axis] = {'range': [None, None]}
                        else:
                            raise RuntimeError('unknown relayout command 2')
                    elif rng == 'showspikes':
                        continue
                    else:
                        raise RuntimeError('unknown relayout command 3')

                except Exception as e:
                    logger().exception(f'Failed to parse relayout_data item k={k}, v={v}; relayout_data={relayout_data}', exc_info=e)
            return layout_dict
        except Exception as e:
            logger().exception(f'Failed to apply relayout_data={relayout_data}', exc_info=e)
    return None


def apply_figure_extent(fig, relayout_data):
    """

    :param fig: plotly.graphical_objects.Figure
    :param relayout_data: dict, e.g. relayout_data={'xaxis.range[0]': '1987-03-28 07:16:06.9794', 'xaxis.range[1]': '2003-01-18 00:53:57.7486'}
    :return: plotly.graphical_objects.Figure
    """
    layout_dict = get_figure_extent(relayout_data)
    print(f'charts.apply_figure_extent: relayout_data={relayout_data}, layout_dict={layout_dict}')
    if isinstance(layout_dict, dict) and layout_dict:
        fig.update_layout(layout_dict)
    return fig
