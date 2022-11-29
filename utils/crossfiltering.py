import numpy as np
import pandas as pd
import xarray as xr
import dash
from dash import html, ALL, ctx, callback
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from utils.graph_with_horizontal_selection_AIO import GraphWithHorizontalSelectionAIO, figure_data_store_id, selected_range_store_id
from utils import charts
import data_processing


SELECT_DATASETS_REQUEST_ID = 'select-datasets-request'
    # 'data' stores a JSON representation of a request executed
FILTER_TAB_CONTAINER_ROW_ID = 'filter-tab-container-row'
    # 'children' contains a layout of the filter tab
FILTER_TYPE_RADIO_ID = 'filter_type_radio'
FILTER_TIME_CONINCIDENCE_SELECT_ID = 'filter_time_coincidence_select'


ds = xr.load_dataset('/home/wolp/data/tmp/aats-sample-merged-timeseries.nc')


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
    )


def get_nbars_slider(i):
    emptry_row = dbc.Row(dbc.Col(html.P()))
    row = dbc.Row(
        [
            dbc.Col(dbc.Label('Number of histogram bars:'), width='auto'),
            dbc.Col(
                dbc.RadioItems(
                    options=[{'label': str(nbars), 'value': nbars} for nbars in [10, 20, 30, 50, 100]],
                    value=50,
                    inline=True,
                    id={'subcomponent': 'nbars_slider', 'aio_id': i},
                ),
                width='auto',
            ),
        ],
        justify='end', #align='baseline',
    )
    return [emptry_row, row]


def get_time_granularity_radio():
    return dbc.InputGroup([
        dbc.InputGroupText('View by: '),
        dbc.RadioItems(
            options=[
                {"label": "year", "value": 'year'},
                {"label": "season", "value": 'season'},
                {"label": "month", "value": 'month'},
            ],
            value='year',
            id={'subcomponent': 'time_granularity_radio', 'aio_id': 'time_filter-time'},
            inline=True,
        ),
    ])


def get_filtering_type_radio():
    simple_vs_cross_filter_radio = dbc.RadioItems(
        options=[
            {'label': 'Simple filter', 'value': 'simple filter'},
            {'label': 'Cross filter', 'value': 'cross filter'},
        ],
        value='simple filter',
        inline=True,
        id=FILTER_TYPE_RADIO_ID,
    )

    time_coincidence_select = dbc.InputGroup(
        [
            dbc.InputGroupText('Observations coincidence time'),
            dbc.Select(
                options=[
                    {'label': '1 hour', 'value': '1H'},
                    {'label': '3 hour', 'value': '3H'},
                    {'label': '6 hour', 'value': '6H'},
                    {'label': '12 hour', 'value': '12H'},
                    {'label': '24 hour', 'value': '24H'},
                    {'label': '48 hour', 'value': '48H'},
                    {'label': '72 hour', 'value': '72H'},
                    {'label': '7 days', 'value': '7D'},
                    {'label': '14 days', 'value': '14D'},
                    {'label': '30 days', 'value': '30D'},
                    {'label': 'custom', 'value': 'custom'},
                ],
                value='24H',
                disabled=True,
                style={'background-color': '#dddddd'},
                id=FILTER_TIME_CONINCIDENCE_SELECT_ID,
            ),
        ],
        id='filter_time_coincidence_select-time_filter-time-tooltip_target',
    )

    tooltip = dbc.Tooltip(
        'Observations of different variables will be considered as coinciding in time if the difference between time measurements does not exceed the selected value',
        target='filter_time_coincidence_select-time_filter-time-tooltip_target',
    )

    cols = [
        dbc.Col(simple_vs_cross_filter_radio, width='auto'),
        dbc.Col([time_coincidence_select, tooltip], width='auto'),
    ]
    return cols


def get_value_by_aio_id(aio_id, ids, values):
    for i, v in zip(ids, values):
        if i['aio_id'] == aio_id:
            return v
    raise ValueError(f'cannot find aio_id={aio_id} among ids={ids}')


def set_value_by_aio_id(aio_id, ids, value):
    return [value if i['aio_id'] == aio_id else dash.no_update for i in ids]


@callback(
    Output(figure_data_store_id(ALL), 'data'),
    Output(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'disabled'),
    Output(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'style'),
    Input(selected_range_store_id(ALL), 'data'),
    Input(selected_range_store_id(ALL), 'id'),
    Input({'subcomponent': 'time_granularity_radio', 'aio_id': ALL}, 'value'),
    Input(FILTER_TYPE_RADIO_ID, 'value'),
    Input(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'value'),
    Input({'subcomponent': 'log_scale_switch', 'aio_id': ALL}, 'value'),
    Input({'subcomponent': 'nbars_slider', 'aio_id': ALL}, 'value'),
    State(figure_data_store_id(ALL), 'id'),
    State({'subcomponent': 'log_scale_switch', 'aio_id': ALL}, 'id'),
    State({'subcomponent': 'nbars_slider', 'aio_id': ALL}, 'id'),
    State(SELECT_DATASETS_REQUEST_ID, 'data'),
    prevent_initial_call=True,
)
def update_histograms_callback(
        selected_ranges, selected_range_ids,
        time_granularity, filter_type, cross_filtering_time_coincidence,
        log_scale_switches, nbars,
        figure_ids, log_scale_switch_ids, nbars_ids,
        select_datasets_request,
):
    if ctx.triggered_id is None or select_datasets_request is None:
        raise PreventUpdate

    cross_filtering = filter_type == 'cross filter'
    if cross_filtering:
        cross_filtering_time_coincidence_dt = pd.Timedelta(cross_filtering_time_coincidence).to_timedelta64()
    else:
        cross_filtering_time_coincidence_dt = None
    filter_time_coincidence_select_style = None if cross_filtering else {'background-color': '#dddddd'}

    req = data_processing.IntegrateDatasetsRequest.from_dict(select_datasets_request)
    ds = req.compute()
    color_mapping = charts.get_color_mapping(ds)

    selected_range_by_aio_id = dict(zip((i['aio_id'] for i in selected_range_ids), selected_ranges))
    rng_by_variable = {}
    for selected_range in selected_range_by_aio_id.values():
        v, x0, x1 = selected_range['variable_label'], selected_range['x_sel_min'], selected_range['x_sel_max']
        if v in rng_by_variable:
            raise RuntimeError(f'variable_label={v} is duplicated among selected_range_ids={selected_range_ids}, selected_ranges={selected_ranges}')
        rng_by_variable[v] = (x0, x1)

    if isinstance(ctx.triggered_id, str) or ctx.triggered_id.get('subcomponent') != 'time_granularity_radio':
        ds_filtered_by_var = data_processing.filter_dataset(
            ds, rng_by_variable,
            cross_filtering=cross_filtering,
            tolerance=cross_filtering_time_coincidence_dt,
        )

    def get_fig(aio_id):
        variable_label = selected_range_by_aio_id[aio_id]['variable_label']

        x_min = ds[variable_label].min().item()
        x_max = ds[variable_label].max().item()

        log_scale_switch = get_value_by_aio_id(aio_id, log_scale_switch_ids, log_scale_switches)
        log_x = 'log_x' in log_scale_switch
        log_y = 'log_y' in log_scale_switch

        bins = get_value_by_aio_id(aio_id, nbars_ids, nbars)

        new_fig = charts.get_histogram(
            ds_filtered_by_var[variable_label],
            variable_label,
            bins=bins,
            color=color_mapping[variable_label],
            x_min=x_min, x_max=x_max,
            log_x=log_x, log_y=log_y
        )
        return {
            'fig': new_fig,
            'rng': [x_min, x_max],
        }

    if ctx.triggered_id in [FILTER_TYPE_RADIO_ID, FILTER_TIME_CONINCIDENCE_SELECT_ID] or \
            not isinstance(ctx.triggered_id, str) and ctx.triggered_id.get('subcomponent') == 'selected_range_store':
        figures_data = []
        for i in figure_ids:
            if i['aio_id'] == 'time_filter-time':
                t_min = pd.Timestamp(min(da['time'].min().values for _, da in ds.items())).strftime('%Y-%m-%d %H:%M')
                t_max = pd.Timestamp(max(da['time'].max().values for _, da in ds.items())).strftime('%Y-%m-%d %H:%M')
                new_fig = {
                    'fig': charts.get_avail_data_by_var_heatmap(
                        data_processing.filter_dataset(
                            ds, rng_by_variable,
                            ignore_time=True,
                            cross_filtering=cross_filtering,
                            tolerance=cross_filtering_time_coincidence_dt,
                        ),
                        time_granularity[0],
                        color_mapping=color_mapping
                    ),
                    'rng': [t_min, t_max],
                }
                figures_data.append(new_fig)
            else:
                figures_data.append(get_fig(i['aio_id']))
        return figures_data, not cross_filtering, filter_time_coincidence_select_style
    elif not isinstance(ctx.triggered_id, str) and ctx.triggered_id.get('subcomponent') in ['log_scale_switch', 'nbars_slider']:
        aio_id = ctx.triggered_id.aio_id
        figure_data = get_fig(aio_id)
        return set_value_by_aio_id(aio_id, figure_ids, figure_data), not cross_filtering, filter_time_coincidence_select_style
    elif not isinstance(ctx.triggered_id, str) and ctx.triggered_id.get('subcomponent') == 'time_granularity_radio':
        t_min = pd.Timestamp(min(da['time'].min().values for _, da in ds.items())).strftime('%Y-%m-%d %H:%M')
        t_max = pd.Timestamp(max(da['time'].max().values for _, da in ds.items())).strftime('%Y-%m-%d %H:%M')
        new_fig = {
            'fig': charts.get_avail_data_by_var_heatmap(
                data_processing.filter_dataset(
                    ds, rng_by_variable,
                    ignore_time=True,
                    cross_filtering=cross_filtering,
                    tolerance=cross_filtering_time_coincidence_dt,
                ),
                time_granularity[0],
                color_mapping=color_mapping
            ),
            'rng': [t_min, t_max],
        }
        return set_value_by_aio_id('time_filter-time', figure_ids, new_fig), not cross_filtering, filter_time_coincidence_select_style
    else:
        raise RuntimeError(f'unknown trigger: {ctx.triggered_id}')


@callback(
    Output(FILTER_TAB_CONTAINER_ROW_ID, 'children'),
    Input(SELECT_DATASETS_REQUEST_ID, 'data'),
    prevent_initial_call=True,
)
def data_filtering_create_layout_callback(select_datasets_request):
    if select_datasets_request is None:
        return None

    req = data_processing.IntegrateDatasetsRequest.from_dict(select_datasets_request)
    ds = req.compute()

    color_mapping = charts.get_color_mapping(ds)

    t_min = pd.Timestamp(min(da['time'].min().values for _, da in ds.items())).strftime('%Y-%m-%d %H:%M')
    t_max = pd.Timestamp(max(da['time'].max().values for _, da in ds.items())).strftime('%Y-%m-%d %H:%M')
    time_filter = GraphWithHorizontalSelectionAIO(
        'time_filter',
        'time',
        variable_label='time',
        x_min=t_min,
        x_max=t_max,
        x_label='time',
        title='Time interval selected:',
        figure=charts.get_avail_data_by_var_heatmap(ds, 'year', color_mapping=color_mapping),
        extra_dash_components=get_time_granularity_radio(),
    )

    filter_and_title_by_v = {'time': (time_filter, 'Data availability')}
    for v, da in ds.items():
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
            extra_dash_components2=get_nbars_slider(f'{v}_filter-scalar'),
        )
        title = da.attrs.get('title', '???')
        city = da.attrs.get('city')
        if city is not None:
            title = f'{title}, {city}'
        filter_and_title_by_v[v] = var_filter, title + f' : {v}'

    return dbc.Accordion(
        [
            dbc.AccordionItem(v_filter, title=title, item_id=f'filter-{v}')
            for v, (v_filter, title) in filter_and_title_by_v.items()
        ],
        always_open=True,
        active_item=[f'filter-{v}' for v in filter_and_title_by_v.keys()]
    )
