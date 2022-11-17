import numpy as np
import pandas as pd
import xarray as xr
from plotly import express as px, graph_objects as go
from plotly.subplots import make_subplots

import data_access

# Color codes
ACTRIS_COLOR_HEX = '#00adb7'
IAGOS_COLOR_HEX = '#456096'
ICOS_COLOR_HEX = '#ec165c'


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


def _get_scatter_plot(ds):
    df = ds.to_dataframe()
    v1, v2, *_ = list(df)
    df = df[[v1, v2]]
    v1_unit = ds[v1].attrs.get('units', '???')
    v2_unit = ds[v2].attrs.get('units', '???')
    fig = px.scatter(df, x=v1, y=v2, height=600)
    return fig


def _get_line_plot(ds):
    df = ds.to_dataframe()
    v1, v2, *_ = list(df)
    df = df[[v1, v2]]
    v1_unit = ds[v1].attrs.get('units', '???')
    v2_unit = ds[v2].attrs.get('units', '???')

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df.index, y=df[v1], name=v1, mode='lines'),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df.index, y=df[v2], name=v2, mode='lines'),
        secondary_y=True,
    )

    # Add figure title
    fig.update_layout(
        title_text=f'{v1} and {v2}',
        height=600
    )

    # Set x-axis title
    fig.update_xaxes(title_text="time")

    # Set y-axes titles
    fig.update_yaxes(title_text=f'{v1} ({v1_unit})', secondary_y=False)
    fig.update_yaxes(title_text=f'{v2} ({v2_unit})', secondary_y=True)

    return fig


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
    import plotly.colors
    list_of_rgb_triples = [plotly.colors.hex_to_rgb(hex_color) for hex_color in px.colors.qualitative.Dark24]
    return list_of_rgb_triples


def get_color_mapping(variables):
    return dict(zip(variables, colors()))


def get_avail_data_by_var_heatmap(ds, granularity, color_mapping=None):
    if color_mapping is None:
        color_mapping = get_color_mapping(ds.data_vars)

    def get_data_avail_with_freq(ds, granularity):
        if granularity == 'year':
            freq = 'YS'
        elif granularity == 'season':
            freq = 'QS-DEC'
        elif granularity == 'month':
            freq = 'MS'
        else:
            raise ValueError(f'unknown granularity={granularity}')
        ds_avail = ds.notnull().resample({'time': freq}).mean()
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

    def get_heatmap(ds_avail, color_mapping):
        n_vars = len(list(ds_avail.data_vars))
        availability_data = np.stack([ds_avail[v].values for i, v in enumerate(ds_avail.data_vars)])
        z_data = availability_data + 2 * np.arange(n_vars).reshape((n_vars, 1))
            # this hook is because we want apply different color scale to each row of availability_data...
        # and here come the color scales:
        colorscale = [
            [[2*i / (2*n_vars), f'rgba{color_mapping[v] + (0,)}'], [(2*i+1) / (2*n_vars), f'rgba{color_mapping[v] + (255,)}']]
            for i, v in enumerate(ds_avail.data_vars)
        ]
        colorscale = sum(colorscale, start=[])
        colorscale.append([1., 'rgba(255, 255, 255, 255)'])  # must define whatever color for z / zmax = 1.
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
            y=list(ds_avail.data_vars),
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
        'height': 120 + 40 * n_vars,
        'margin': {'b': 80, 't': 40},
    }

    fig = go.Figure(data=get_heatmap(ds_avail, color_mapping), layout=layout_dict)
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
    # print(ds_avail['time'])
    return fig


def get_avail_data_by_var_heatmap_old(ds, granularity, color_mapping=None):
    if color_mapping is None:
        color_mapping = get_color_mapping(ds.data_vars)

    def get_data_avail_with_freq(ds, granularity):
        if granularity == 'year':
            freq = 'YS'
        elif granularity == 'season':
            freq = 'QS-DEC'
        elif granularity == 'month':
            freq = 'MS'
        else:
            raise ValueError(f'unknown granularity={granularity}')
        ds_avail = ds.notnull().resample({'time': freq}).mean()
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
        ds_avail['time2'] = t2
        return ds_avail.set_coords('time2')

    def get_heatmap(ds_avail, v, color):
        heatmap = go.Heatmap(
            z=np.expand_dims(ds_avail[v].values, 0),
            x=ds_avail['time2'],
            y=[v],
            colorscale=[[0., f'rgba{color + (0, )}'], [1., f'rgba{color + (255, )}']],
            hovertemplate='%{x}: %{z}',
            name=v,
            showscale=False,
            xgap=1,
        )
        return heatmap

    ds_avail = get_data_avail_with_freq(ds, granularity)
    traces = [get_heatmap(ds_avail, v, color) for v, color in color_mapping.items()]
    for i, trace in enumerate(reversed(traces)):
        if i > 0:
            trace.update(yaxis=f'y{i+1}')

    n_vars = max(len(traces), 1)
    dy = 1 / n_vars
    ys = np.linspace(0, 1, n_vars + 1)
    layout_dict = {
        'autosize': False,
        'height': 100 + 40 * n_vars,
        'margin': {'b': 80, 't': 40},
    }
    for i, y1, y2 in zip(range(1, n_vars + 1), ys[:-1], ys[1:]):
        yaxis = 'yaxis' if i == 1 else f'yaxis{i}'
        layout_dict[yaxis] = {'domain': [y1 + 0.05 * dy, y2 - 0.05 * dy]}
    layout = go.Layout(layout_dict)
    fig = go.Figure(data=traces, layout=layout)
    fig.update_xaxes(title='time')
    return fig


def get_histogram(da, x_label, bins=20, color=None, x_min=None, x_max=None, log_x=False, log_y=False):
    ar = da.where(da.notnull(), drop=True).values
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
    fig = go.Figure(data=[
        go.Bar(
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
            marker={'color': f'rgb{color}' if isinstance(color, tuple) and len(color) == 3 else color}
        )
    ])
    fig.update_layout(
        title=da.attrs['long_name'],
        xaxis_title=f"{da.attrs['standard_name']} ({da.attrs['units']})",
        yaxis_title='# observations',
    )

    if log_x:
        fig.update_xaxes(type='log')
    if log_y:
        fig.update_yaxes(type='log')

    return fig


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
