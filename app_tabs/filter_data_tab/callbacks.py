import numpy as np
import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import callback, Output, ALL, Input, State, ctx
from dash.exceptions import PreventUpdate

import data_processing
from data_processing import metadata
from .layout import FILTER_TIME_CONINCIDENCE_SELECT_ID, FILTER_TYPE_RADIO_ID, \
    FILTER_TAB_CONTAINER_ROW_ID, FILTER_DATA_BUTTON_ID, \
    get_time_granularity_radio, get_log_axis_switches, get_nbars_slider
from app_tabs.common.callbacks import get_value_by_aio_id, set_value_by_aio_id
from app_tabs.common.layout import INTEGRATE_DATASETS_REQUEST_ID, FILTER_DATA_REQUEST_ID
from utils import charts
from utils.graph_with_horizontal_selection_AIO import figure_data_store_id, selected_range_store_id, \
    GraphWithHorizontalSelectionAIO
from log import log_exception


def _get_min_max_time(da_by_var):
    t_min, t_max = None, None
    for _, da in da_by_var.items():
        t = da['time']
        if len(t) == 0:
            continue
        t0, t1 = t.min().values, t.max().values
        if not np.isnan(t0) and t_min is None or t0 < t_min:
            t_min = t0
        if not np.isnan(t1) and t_max is None or t1 > t_max:
            t_max = t1
    if t_min is not None:
        t_min = pd.Timestamp(t_min).strftime('%Y-%m-%d %H:%M')
    if t_max is not None:
        t_max = pd.Timestamp(t_max).strftime('%Y-%m-%d %H:%M')
    return t_min, t_max


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
    State(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
def update_histograms_callback(
        selected_ranges, selected_range_ids,
        time_granularity, filter_type, cross_filtering_time_coincidence,
        log_scale_switches, nbars,
        figure_ids, log_scale_switch_ids, nbars_ids,
        integrate_datasets_request,
):
    if ctx.triggered_id is None or integrate_datasets_request is None:
        raise PreventUpdate

    cross_filtering = filter_type == 'cross filter'
    if cross_filtering:
        cross_filtering_time_coincidence_dt = pd.Timedelta(cross_filtering_time_coincidence).to_timedelta64()
    else:
        cross_filtering_time_coincidence_dt = None
    filter_time_coincidence_select_style = None if cross_filtering else {'background-color': '#dddddd'}

    req = data_processing.IntegrateDatasetsRequest.from_dict(integrate_datasets_request)
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
                t_min, t_max = _get_min_max_time(ds)
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
        t_min, t_max = _get_min_max_time(ds)
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
    Input(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
def data_filtering_create_layout_callback(integrate_datasets_request):
    if integrate_datasets_request is None:
        return None

    req = data_processing.IntegrateDatasetsRequest.from_dict(integrate_datasets_request)
    ds = req.compute()

    color_mapping = charts.get_color_mapping(ds)

    t_min, t_max = _get_min_max_time(ds)

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

        md = metadata.da_attr_to_metadata_dict(da=da)
        title = md[metadata.VARIABLE_LABEL]
        city_or_station_name = md[metadata.CITY_OR_STATION_NAME]
        if city_or_station_name is not None:
            title = f'{title}, {city_or_station_name}'
        filter_and_title_by_v[v] = var_filter, f'{v} : {title}'

    # TODO: maybe the accordion should go to layout ???
    return dbc.Accordion(
        [
            dbc.AccordionItem(v_filter, title=title, item_id=f'filter-{v}')
            for v, (v_filter, title) in filter_and_title_by_v.items()
        ],
        always_open=True,
        active_item=[f'filter-{v}' for v in filter_and_title_by_v.keys()],
        # style={'text-transform': None},
    )


# TODO: lots of duplications with utils.crossfiltering.update_histograms_callback
@callback(
    Output(FILTER_DATA_REQUEST_ID, 'data'),
    Input(FILTER_DATA_BUTTON_ID, 'n_clicks'),
    State(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    State(selected_range_store_id(ALL), 'data'),
    State(selected_range_store_id(ALL), 'id'),
    State(FILTER_TYPE_RADIO_ID, 'value'),
    State(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'value'),
    prevent_initial_call=True,
)
@log_exception
def filter_data_callback(
        n_clicks,
        integrate_datasets_request,
        selected_ranges, selected_range_ids,
        filter_type, cross_filtering_time_coincidence
):
    if dash.ctx.triggered_id is None or integrate_datasets_request is None:
        raise dash.exceptions.PreventUpdate

    # TODO: this is a duplication with utils.crossfiltering.update_histograms_callback
    cross_filtering = filter_type == 'cross filter'
    if cross_filtering:
        cross_filtering_time_coincidence_dt = pd.Timedelta(cross_filtering_time_coincidence).to_timedelta64()
    else:
        cross_filtering_time_coincidence_dt = None

    # TODO: this is a duplication with utils.crossfiltering.update_histograms_callback
    integrate_datasets_req = data_processing.IntegrateDatasetsRequest.from_dict(integrate_datasets_request)

    # TODO: this is a duplication with utils.crossfiltering.update_histograms_callback
    selected_range_by_aio_id = dict(zip((i['aio_id'] for i in selected_range_ids), selected_ranges))
    rng_by_varlabel = {}
    for selected_range in selected_range_by_aio_id.values():
        v, x0, x1 = selected_range['variable_label'], selected_range['x_sel_min'], selected_range['x_sel_max']
        if v in rng_by_varlabel:
            raise RuntimeError(f'variable_label={v} is duplicated among selected_range_ids={selected_range_ids}, selected_ranges={selected_ranges}')
        rng_by_varlabel[v] = (x0, x1)

    filter_data_req = data_processing.FilterDataRequest(
        integrate_datasets_req,
        rng_by_varlabel,
        cross_filtering,
        cross_filtering_time_coincidence_dt
    )

    filter_data_req.compute()

    return filter_data_req.to_dict()
