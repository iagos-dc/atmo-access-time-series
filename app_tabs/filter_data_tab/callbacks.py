import warnings
import numpy as np
import dash
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc
import pandas as pd
from dash import callback, Output, ALL, Input, State, ctx, html
from dash.exceptions import PreventUpdate

import data_processing
from data_processing import metadata
from app_tabs.common.layout import FILTER_DATA_TAB_VALUE, APP_TABS_ID, DATA_ANALYSIS_TAB_VALUE
from .layout import FILTER_TIME_CONINCIDENCE_INPUTGROUP_ID, FILTER_TIME_CONINCIDENCE_SELECT_ID, FILTER_TYPE_RADIO_ID, \
    TIME_FILTER_CONTAINER_ID, VARIABLE_FILTERS_CONTAINER_ID, FILTER_DATA_BUTTON_ID, \
    get_time_granularity_radio, get_log_axis_switches, get_nbars_slider
from app_tabs.common.callbacks import get_value_by_aio_id, set_value_by_aio_id
from app_tabs.common.layout import INTEGRATE_DATASETS_REQUEST_ID, FILTER_DATA_REQUEST_ID, get_tooltip
from utils import charts
from utils.graph_with_horizontal_selection_AIO import figure_data_store_id, selected_range_store_id, \
    GraphWithHorizontalSelectionAIO, graph_id, interval_input_group_id
from utils.exception_handler import callback_with_exc_handling, AppException, AppWarning


DATA_FILTER_AIO_CLASS = 'data_filter'


@callback_with_exc_handling(
    Output(FILTER_DATA_TAB_VALUE, 'disabled'),
    Input(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
)
def enable_filter_data_tab(integrate_datasets_request):
    return integrate_datasets_request is None


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


@callback_with_exc_handling(
    Output(figure_data_store_id(ALL, DATA_FILTER_AIO_CLASS), 'data'),
    Output(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'disabled'),
    Output(FILTER_TIME_CONINCIDENCE_INPUTGROUP_ID, 'style'),
    #Output(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'style'),
    Input(selected_range_store_id(ALL, DATA_FILTER_AIO_CLASS), 'data'),
    Input(selected_range_store_id(ALL, DATA_FILTER_AIO_CLASS), 'id'),
    Input({'subcomponent': 'time_granularity_radio', 'aio_id': ALL}, 'value'),
    Input(FILTER_TYPE_RADIO_ID, 'value'),
    Input(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'value'),
    Input({'subcomponent': 'log_scale_switch', 'aio_id': ALL}, 'value'),
    Input({'subcomponent': 'nbars_slider', 'aio_id': ALL}, 'value'),
    State(figure_data_store_id(ALL, DATA_FILTER_AIO_CLASS), 'id'),
    State({'subcomponent': 'log_scale_switch', 'aio_id': ALL}, 'id'),
    State({'subcomponent': 'nbars_slider', 'aio_id': ALL}, 'id'),
    State(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    prevent_initial_call=True,
)
def update_histograms_callback(
        selected_ranges, selected_range_ids,
        time_granularity, filter_type, cross_filtering_time_coincidence,
        log_scale_switches, nbars,
        figure_ids, log_scale_switch_ids, nbars_ids,
        integrate_datasets_request,
):
    # TODO: use dash_ctx instead of ctx.triggered_id
    # dash_ctx = list(dash.ctx.triggered_prop_ids.values())
    # print(f'update_histograms_callback::dash_ctx={dash_ctx}')

    if ctx.triggered_id is None or integrate_datasets_request is None:
        raise PreventUpdate

    cross_filtering = filter_type == 'cross filter'
    if cross_filtering:
        cross_filtering_time_coincidence_dt = pd.Timedelta(cross_filtering_time_coincidence).to_timedelta64()
    else:
        cross_filtering_time_coincidence_dt = None
    #filter_time_coincidence_select_style = None if cross_filtering else {'background-color': '#dddddd'} #{'display': 'none'}
    filter_time_coincidence_select_style = None if cross_filtering else {'display': 'none'}

    req = data_processing.IntegrateDatasetsRequest.from_dict(integrate_datasets_request)
    integrate_datasets_request_hash = req.deterministic_hash()
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
    else:
        # TODO: this is a dirty patch; do it properly as indicated in the comment in the l. 132
        ds_filtered_by_var = {}

    def get_fig(aio_id):
        variable_label = selected_range_by_aio_id[aio_id]['variable_label']

        da = ds[variable_label]
        if len(da) > 0:
            x_min = da.min().item()
            x_max = da.max().item()
        else:
            x_min, x_max = None, None

        log_scale_switch = get_value_by_aio_id(aio_id, log_scale_switch_ids, log_scale_switches)
        log_x = 'log_x' in log_scale_switch
        log_y = 'log_y' in log_scale_switch

        bins = get_value_by_aio_id(aio_id, nbars_ids, nbars)

        new_fig, y_max = charts.get_histogram(
            # TODO: is ds_filtered_by_var always set? check if the condition in l. 105 implies the condition from l.148&152|176
            ds_filtered_by_var.get(variable_label),
            variable_label,
            bins=bins,
            color=color_mapping[variable_label],
            x_min=x_min, x_max=x_max,
            log_x=log_x, log_y=log_y
        )
        new_fig = new_fig.update_layout(
            title=f'Distribution of {variable_label}',
            uirevision=integrate_datasets_request_hash + f':{log_x}'
        )

        return {
            'fig': new_fig,
            'rng': [x_min, x_max],
            'rng_y': [0, y_max]
        }

    if ctx.triggered_id in [FILTER_TYPE_RADIO_ID, FILTER_TIME_CONINCIDENCE_SELECT_ID] or \
            not isinstance(ctx.triggered_id, str) and ctx.triggered_id.get('subcomponent') == 'selected_range_store':
        figures_data = []
        for i in figure_ids:
            if i['aio_id'] == 'time_filter-time':
                t_min, t_max = _get_min_max_time(ds)

                fig = charts.get_avail_data_by_var_heatmap(
                    data_processing.filter_dataset(
                        ds, rng_by_variable,
                        ignore_time=True,
                        cross_filtering=cross_filtering,
                        tolerance=cross_filtering_time_coincidence_dt,
                    ),
                    time_granularity[0],
                    color_mapping=color_mapping
                )
                fig = fig.update_layout(
                    title='Data availability',
                    uirevision=integrate_datasets_request_hash
                )

                new_fig = {
                    'fig': fig,
                    'rng': [t_min, t_max],
                    'rng_y': [-0.5, len(ds) - 0.5]
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

        fig = charts.get_avail_data_by_var_heatmap(
            data_processing.filter_dataset(
                ds, rng_by_variable,
                ignore_time=True,
                cross_filtering=cross_filtering,
                tolerance=cross_filtering_time_coincidence_dt,
            ),
            time_granularity[0],
            color_mapping=color_mapping
        )
        fig = fig.update_layout(
            title='Data availability',
            uirevision=integrate_datasets_request_hash
        )

        new_fig = {
            'fig': fig,
            'rng': [t_min, t_max],
            'rng_y': [-0.5, len(ds) - 0.5]
        }
        return set_value_by_aio_id('time_filter-time', figure_ids, new_fig), not cross_filtering, filter_time_coincidence_select_style
    else:
        raise RuntimeError(f'unknown trigger: {ctx.triggered_id}')


def _get_filter_container(v, v_filter):
    aio_id = f'{v}_filter-' + ('scalar' if v != 'time' else 'time')
    range_controller_tooltip = get_tooltip(
        f'Set up a filter on {v} by providing min and/or max thresholds',
        interval_input_group_id(aio_id, DATA_FILTER_AIO_CLASS)
    )

    graph_tooltip = get_tooltip(
        f'Drag-and-drop to set up a filter on {v}',
        graph_id(aio_id, DATA_FILTER_AIO_CLASS)
    )

    data_stores = v_filter.get_data_stores()
    range_controller = v_filter.get_range_controller()
    graph = v_filter.get_graph()
    accordion_item_children = html.Div( #dbc.Container(
        [
            data_stores,
            dbc.Row(
                [
                    dbc.Col(
                        [range_controller, range_controller_tooltip],
                        width=4,
                        align='start',
                    ),
                    dbc.Col(
                        [
                            #graph,
                            dbc.Card(dbc.CardBody(graph)),
                            graph_tooltip
                        ],
                        width=8,
                        align='start',
                    ),
                ],
                align='start',
            ),
        ],
        #fluid=True,
    )
    return accordion_item_children


def _get_accordion(list_of_v_title_component):
    accordion_items = []
    for i, (v, title, component) in enumerate(list_of_v_title_component):
        accordion_items.append(
            dmc.AccordionItem(
                children=[
                    dmc.AccordionControl(title, style={'text-transform': 'none'}),
                    dmc.AccordionPanel(component)
                ],
                value=f'accordion-item-{i}',
            )
        )

    return dmc.Accordion(
        children=accordion_items,
        chevronPosition='left',
        variant='contained',
        radius='lg',
        value=[f'accordion-item-{i}' for i in range(len(list_of_v_title_component))],
        multiple=True,
    )


@callback_with_exc_handling(
    Output(TIME_FILTER_CONTAINER_ID, 'children'),
    Output(VARIABLE_FILTERS_CONTAINER_ID, 'children'),
    Input(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    Input(APP_TABS_ID, 'active_tab'),  # dummy trigger; it is a way to workaround plotly bug of badly resized figures
    prevent_initial_call=True,
)
def data_filtering_create_layout_callback(integrate_datasets_request, app_tab_value):
    if app_tab_value != FILTER_DATA_TAB_VALUE:
        raise PreventUpdate

    if integrate_datasets_request is None:
        return []  # children=None instead of [] does not work

    req = data_processing.IntegrateDatasetsRequest.from_dict(integrate_datasets_request)
    ds = req.compute()
    if len(ds) == 0:
        warnings.warn('No variables found. Choose another dataset(s).', category=AppWarning)
        return []  # children=None does not work

    color_mapping = charts.get_color_mapping(ds)

    t_min, t_max = _get_min_max_time(ds)

    avail_data_by_var_heatmap = charts.get_avail_data_by_var_heatmap(ds, 'year', color_mapping=color_mapping)
    avail_data_by_var_heatmap = avail_data_by_var_heatmap.update_layout(title='Data availability')

    time_filter = GraphWithHorizontalSelectionAIO(
        aio_id='time_filter',
        aio_class=DATA_FILTER_AIO_CLASS,
        x_axis_type='time',
        variable_label='time',
        x_min=t_min,
        x_max=t_max,
        x_label='time',
        figure=avail_data_by_var_heatmap,
        extra_dash_components=get_time_granularity_radio(),
    )

    filter_and_title_by_v = {}
    for v, da in ds.items():
        x_min = da.min().item()
        x_max = da.max().item()

        histogram_fig, _ = charts.get_histogram(da, v, color=color_mapping[v], x_min=x_min, x_max=x_max)
        histogram_fig = histogram_fig.update_layout(title=f'Distribution of {v}')

        var_filter = GraphWithHorizontalSelectionAIO(
            aio_id=f'{v}_filter',
            aio_class=DATA_FILTER_AIO_CLASS,
            x_axis_type='scalar',
            variable_label=v,
            x_min=x_min,
            x_max=x_max,
            x_label=v,
            figure=histogram_fig,
            extra_dash_components=get_log_axis_switches(f'{v}_filter-scalar'),
            extra_dash_components2=get_nbars_slider(f'{v}_filter-scalar'),
        )

        md = metadata.da_attr_to_metadata_dict(da=da)
        title = md[metadata.VARIABLE_LABEL]
        city_or_station_name = md[metadata.CITY_OR_STATION_NAME]
        if city_or_station_name is not None:
            title = f'{title}, {city_or_station_name}'
        filter_and_title_by_v[v] = (
            var_filter,
            dmc.Group([
                html.B(dmc.Text(v)),
                dmc.Text(title, size='sm', fw=400, c='dimmed'),
            ])
        )

    time_filter_accordion = _get_accordion([
        (
            'time',
            dmc.Group([
                html.B(dmc.Text('Time')),
            ]),
            _get_filter_container('time', time_filter)
        )
    ])

    variable_filters_accordion_items_children = []
    for v, (v_filter, title) in filter_and_title_by_v.items():
        variable_filters_accordion_items_children.append(
            (
                v,
                title,
                _get_filter_container(v, v_filter)
            )
        )
    variable_filters_accordion = _get_accordion(variable_filters_accordion_items_children)

    return time_filter_accordion, variable_filters_accordion


# TODO: lots of duplications with utils.crossfiltering.update_histograms_callback
@callback_with_exc_handling(
    Output(FILTER_DATA_REQUEST_ID, 'data'),
    Output(APP_TABS_ID, 'active_tab', allow_duplicate=True),
    Input(FILTER_DATA_BUTTON_ID, 'n_clicks'),
    State(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    State(selected_range_store_id(ALL, DATA_FILTER_AIO_CLASS), 'data'),
    State(selected_range_store_id(ALL, DATA_FILTER_AIO_CLASS), 'id'),
    State(FILTER_TYPE_RADIO_ID, 'value'),
    State(FILTER_TIME_CONINCIDENCE_SELECT_ID, 'value'),
    prevent_initial_call=True,
)
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

    da_filtered_by_var = filter_data_req.compute(store_request=True)
    if len(da_filtered_by_var) == 0:
        warnings.warn(
            'The applied data filter is too restrictive: all variables\' values were masked out. '
            'Please change the filter.',
            category=AppWarning
        )
        next_tab = dash.no_update
    else:
        next_tab = DATA_ANALYSIS_TAB_VALUE

    return filter_data_req.to_dict(), next_tab
