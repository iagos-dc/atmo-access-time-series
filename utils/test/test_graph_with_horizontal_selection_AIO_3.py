import numpy as np
import pandas as pd
import xarray as xr
import toolz
import plotly.express as px
import dash
from dash import Dash, html, MATCH, ALL, ctx, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from utils.graph_with_horizontal_selection_AIO import GraphWithHorizontalSelectionAIO, figure_data_store_id, selected_range_store_id
from utils import charts


ds = xr.load_dataset('/home/wolp/data/tmp/aats-sample-merged-timeseries.nc')

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
        #'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
)


def get_log_axis_switches(i):
    return dbc.Checklist(
        options=[
            {'label': 'x-axis in log-scale', 'value': 'log_x'},
            {'label': 'y-axis in log-scale', 'value': 'log_y'},
        ],
        value=[],
        id={'subcomponent': 'log_scale_switch', 'aio_id': i},
        inline=True,
        switch=True,
        # size='sm',
        # className='mb-3',
    )


def get_time_granularity_radio():
    return dbc.RadioItems(
        options=[
            {"label": "year", "value": 'year'},
            {"label": "season", "value": 'season'},
            {"label": "month", "value": 'month'},
        ],
        value='year',
        #id='time_granularity_radio',
        id={'subcomponent2': 'time_granularity_radio', 'foo': 'bar'},
        inline=True,
    ),


# graph_with_horizontal_selection_AIO = GraphWithHorizontalSelectionAIO(
#     'foo',
#     'scalar',
#     x_min=x_min,
#     x_max=x_max,
#     x_label='O3',
#     title='Time interval selected:',
#     extra_dash_components=log_axis_switches
#     #figure=charts.get_avail_data_by_var(ds),
#     #figure=fig,
#     #x_min=min(x).strftime('%Y-%m-%d %H:%M'),
#     #x_max=max(x).strftime('%Y-%m-%d %H:%M'),
# )


def get_value_by_aio_id(aio_id, ids, values):
    for i, v in zip(ids, values):
        if i['aio_id'] == aio_id:
            return v
    raise ValueError(f'cannot find aio_id={aio_id} among ids={ids}')


def set_value_by_aio_id(aio_id, ids, value):
    return [value if i['aio_id'] == aio_id else dash.no_update for i in ids]


def filter_dataset(ds, rng_by_variable, ignore_time=False):
    def get_cond_conjunction(conds):
        cond_conjunction = True
        for cond in conds:
            cond_conjunction &= cond
        return cond_conjunction

    if ignore_time:
        rng_by_variable = rng_by_variable.copy()
        rng_by_variable.pop('time', None)

    cond_by_variable = {}
    for v, rng in rng_by_variable.items():
        _min, _max = rng
        if isinstance(_min, str):
            _min = np.datetime64(_min)
        if isinstance(_max, str):
            _max = np.datetime64(_max)
        cond = ds[v].notnull() | ~ds[v].notnull()
        if _min is not None:
            cond &= (ds[v] >= _min)
        if _max is not None:
            cond &= (ds[v] <= _max)
        cond_by_variable[v] = cond
        print(f'v={v}, rng={rng}, cond={cond_by_variable[v].sum().item()}')

    ds_filtered = {}
    for v in ds.data_vars:
        if not ignore_time:
            conds = [cond for v_other, cond in cond_by_variable.items() if v_other != v]
        else:
            conds = cond_by_variable.values()
        ds_filtered[v] = ds[v].where(get_cond_conjunction(conds), drop=False)
    ds_filtered = xr.Dataset(ds_filtered)
    return ds_filtered


def filter_dataset_old(ds, rng_by_variable):
    cond_by_variable = {}
    for v, rng in rng_by_variable.items():
        _min, _max = rng
        if isinstance(_min, str):
            _min = np.datetime64(_min)
        if isinstance(_max, str):
            _max = np.datetime64(_max)
        cond = ds[v].notnull() | ~ds[v].notnull()
        if _min is not None: # and not np.isnan(_min):
            cond &= ds[v] >= _min
            # cond &= ds[v].isnull() | (ds[v] >= _min)
        if _max is not None: # and not np.isnan(_max):
            cond &= ds[v] <= _max
            # cond &= ds[v].isnull() | (ds[v] <= _max)
        cond_by_variable[v] = cond

    if len(cond_by_variable) > 0:
        cond_iter = iter(cond_by_variable.values())
        total_cond = next(cond_iter)
        for cond in cond_iter:
            total_cond &= cond
        return ds.where(total_cond, drop=False)
    else:
        return ds


@callback(
    Output(figure_data_store_id(ALL), 'data'),
    Input(selected_range_store_id(ALL), 'data'),
    Input(selected_range_store_id(ALL), 'id'),
    Input({'subcomponent': 'log_scale_switch', 'aio_id': ALL}, 'value'),
    Input({'subcomponent2': 'time_granularity_radio', 'foo': ALL}, 'value'),
    State(figure_data_store_id(ALL), 'id'),
    State({'subcomponent': 'log_scale_switch', 'aio_id': ALL}, 'id'),
)
def update_histograms_callback(selected_ranges, selected_range_ids, log_scale_switches, time_granularity, figure_ids, log_scale_switch_ids):
    #print(f'figure_ids={figure_ids}')
    #print(f'selected_ranges={selected_ranges}')
    #print(f'selected_ranges={selected_range_ids}')
    #print(f'log_scale_switches={log_scale_switches}')
    #print(f'log_scale_switch_ids={log_scale_switch_ids}')
    if ctx.triggered_id is None:
        raise PreventUpdate

    selected_range_by_aio_id = dict(zip((i['aio_id'] for i in selected_range_ids), selected_ranges))
    rng_by_variable = {}
    for selected_range in selected_range_by_aio_id.values():
        v, x0, x1 = selected_range['variable_label'], selected_range['x_sel_min'], selected_range['x_sel_max']
        if v in rng_by_variable:
            raise RuntimeError(f'variable_label={v} is duplicated among selected_range_ids={selected_range_ids}, selected_ranges={selected_ranges}')
        rng_by_variable[v] = (x0, x1)

    ds_filtered = filter_dataset_old(ds, rng_by_variable)
    ds_filtered_by_var = filter_dataset(ds, rng_by_variable)
    color_mapping = charts.get_color_mapping(ds.data_vars)

    def get_fig(aio_id):
        variable_label = selected_range_by_aio_id[aio_id]['variable_label']

        # x_min = ds_filtered[variable_label].min().item()
        # x_max = ds_filtered[variable_label].max().item()
        x_min = ds[variable_label].min().item()
        x_max = ds[variable_label].max().item()

        log_scale_switch = get_value_by_aio_id(aio_id, log_scale_switch_ids, log_scale_switches)
        log_x = 'log_x' in log_scale_switch
        log_y = 'log_y' in log_scale_switch

        new_fig = charts.get_histogram(ds_filtered_by_var[variable_label], variable_label, color=color_mapping[variable_label], x_min=x_min, x_max=x_max, log_x= log_x, log_y=log_y)
        return {
            'fig': new_fig,
            'rng': [x_min, x_max],
        }

    if ctx.triggered_id.get('subcomponent') == 'selected_range_store':
        figures_data = []
        for i in figure_ids:
            if i['aio_id'] == 'time_filter-time':
                # figures_data.append(dash.no_update)
                t_min = pd.Timestamp(ds.time.min().values).strftime('%Y-%m-%d %H:%M')
                t_max = pd.Timestamp(ds.time.max().values).strftime('%Y-%m-%d %H:%M')
                new_fig = {
                    'fig': charts.get_avail_data_by_var_heatmap(
                        filter_dataset(ds, rng_by_variable, ignore_time=True),
                        time_granularity[0],
                        color_mapping=color_mapping
                    ),
                    'rng': [t_min, t_max],
                }
                figures_data.append(new_fig)
            else:
                figures_data.append(get_fig(i['aio_id']))
        return figures_data
    elif ctx.triggered_id.get('subcomponent') == 'log_scale_switch':
        aio_id = ctx.triggered_id.aio_id
        figure_data = get_fig(aio_id)
        return set_value_by_aio_id(aio_id, figure_ids, figure_data)
    elif ctx.triggered_id.get('subcomponent2') == 'time_granularity_radio':
        t_min = pd.Timestamp(ds.time.min().values).strftime('%Y-%m-%d %H:%M')
        t_max = pd.Timestamp(ds.time.max().values).strftime('%Y-%m-%d %H:%M')
        new_fig = {
            'fig': charts.get_avail_data_by_var_heatmap(
                filter_dataset(ds, rng_by_variable, ignore_time=True),
                time_granularity[0],
                color_mapping=color_mapping
            ),
            'rng': [t_min, t_max],
        }
        return set_value_by_aio_id('time_filter-time', figure_ids, new_fig)
    else:
        raise RuntimeError(f'unknown trigger: {ctx.triggered_id}')


app.layout = html.Div(
    [
        # all_selected_ranges_store(),
        dbc.ListGroup(id='list-group'),
        dbc.Button('Go!', id='go_fig_button_id', n_clicks=0, type='submit')
    ]
)


@callback(
    Output('list-group', 'children'),
    Input('go_fig_button_id', 'n_clicks'),
)
def go_callback(go_fig_buttion_n_clicks):
    color_mapping = charts.get_color_mapping(ds.data_vars)

    t_min = pd.Timestamp(ds.time.min().values).strftime('%Y-%m-%d %H:%M')
    t_max = pd.Timestamp(ds.time.max().values).strftime('%Y-%m-%d %H:%M')
    time_filter = GraphWithHorizontalSelectionAIO(
        'time_filter',
        'time',
        variable_label='time',
        x_min=t_min,
        x_max=t_max,
        x_label='time',
        title='Time interval selected:',
        #figure=charts.get_avail_data_by_var_gantt(ds),
        figure=charts.get_avail_data_by_var_heatmap(ds, 'year', color_mapping=color_mapping),
        extra_dash_components=get_time_granularity_radio(),
    )

    var_filters = []
    for v in sorted(ds.data_vars):
        da = ds[v]
        x_min = da.min().item()
        x_max = da.max().item()

        var_filter = GraphWithHorizontalSelectionAIO(
            f'{v}_filter',
            'scalar',
            variable_label=v,
            x_min=x_min,
            x_max=x_max,
            x_label=v,
            title=f'{v} interval selected:',
            figure=charts.get_histogram(da, v, color=color_mapping[v], x_min=x_min, x_max=x_max),
            extra_dash_components=get_log_axis_switches(f'{v}_filter-scalar'),
        )
        var_filters.append(var_filter)

    return dbc.ListGroup([time_filter] + var_filters)


if __name__ == "__main__":
    app.run_server(debug=True, host='localhost', port=8055)
