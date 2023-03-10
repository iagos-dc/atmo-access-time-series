import numpy as np
import pandas as pd
import dash
import toolz
from dash import Input, Output, ALL
import dash_bootstrap_components as dbc

import data_processing
from data_processing import metadata, analysis
from app_tabs.common.layout import FILTER_DATA_REQUEST_ID
from . import common_layout
from app_tabs.data_analysis_tab import trend_analysis_layout
from log import log_exception, print_callback
from utils import dash_dynamic_components as ddc, charts, dash_persistence, helper
from utils.broadcast import broadcast
from utils.graph_with_horizontal_selection_AIO import figure_data_store_id, selected_range_store_id


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


@ddc.dynamic_callback(
    ddc.DynamicOutput(trend_analysis_layout.AGGREGATE_COLLAPSE_ID, 'is_open'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATE_CHECKBOX_ID, 'value'),
)
@log_exception
def show_aggregate_card(aggregate_checkbox):
    return aggregate_checkbox


@ddc.dynamic_callback(
    ddc.DynamicOutput(trend_analysis_layout.DESEASONIZE_COLLAPSE_ID, 'is_open'),
    ddc.DynamicInput(trend_analysis_layout.DESEASONIZE_CHECKBOX_ID, 'value'),
)
@log_exception
def show_deseasonize_card(deseasonize_checkbox):
    return deseasonize_checkbox


@ddc.dynamic_callback(
    ddc.DynamicOutput(trend_analysis_layout.TREND_SUMMARY_CONTAINER_ID, 'children'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
)
@log_exception
@print_callback()
def get_trend_summary(
        filter_data_request,
):
    if helper.any_is_None(filter_data_request):
        raise dash.exceptions.PreventUpdate

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()
    da_by_var = filter_data_request.compute()
    # colors_by_var = charts.get_color_mapping(da_by_var)

    if len(da_by_var) == 0:
        return None

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)
    variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    yaxis_label_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    children = []
    for v, da in da_by_var.items():
        md = metadata_by_var.get(v, {})
        (a, b), (ci0, ci1), (x_unit, y_unit) = data_processing.analysis.theil_sen_slope(da.to_series(), subsampling=3000)
        a = a * y_unit * (np.timedelta64(365 * 24 * 3600, 's') / x_unit)
        ci0 = ci0 * y_unit * (np.timedelta64(365 * 24 * 3600, 's') / x_unit)
        ci1 = ci1 * y_unit * (np.timedelta64(365 * 24 * 3600, 's') / x_unit)
        units = md.get(metadata.UNITS, '???')
        trend_summary = f'{v} : {a:.4g} {units} / year; 95% CI: [{ci0:.4g}, {ci1:.4g}] {units} / year'
        children.append(dbc.Row(trend_summary))

    return children


@ddc.dynamic_callback(
    Output(figure_data_store_id(trend_analysis_layout.TREND_ANALYSIS_AIO_ID + '-time', trend_analysis_layout.TREND_ANALYSIS_AIO_CLASS), 'data'),
    ddc.DynamicOutput(trend_analysis_layout.TREND_GRAPH_ID, 'figure'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(common_layout.DATA_ANALYSIS_VARIABLES_CHECKLIST_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.TREND_ANALYSIS_METHOD_RADIO_ID, 'value'),
    Input(selected_range_store_id(trend_analysis_layout.TREND_ANALYSIS_AIO_ID + '-time', trend_analysis_layout.TREND_ANALYSIS_AIO_CLASS), 'data'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATE_CHECKBOX_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATION_PERIOD_SELECT_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATION_FUNCTION_SELECT_ID, 'value'),
    ddc.DynamicInput(common_layout.MIN_SAMPLE_SIZE_INPUT_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.DESEASONIZE_CHECKBOX_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.MOVING_AVERAGE_PERIOD_SELECT_ID, 'value')
)
@log_exception
@print_callback()
def get_trend_plots_callback(
        filter_data_request,
        vs,
        analysis_method,
        time_rng,
        do_aggregate,
        aggregation_period,
        aggregation_function,
        min_sample_size,
        do_deseasonize,
        moving_average_period
):
    dash_ctx = list(dash.ctx.triggered_prop_ids.values())
    print(f'get_trend_plot_callback dash_ctx={dash_ctx}')

    if helper.any_is_None(
        filter_data_request,
        vs,
        analysis_method,
        do_aggregate, aggregation_period, aggregation_function, min_sample_size,
        do_deseasonize, moving_average_period
    ):
        raise dash.exceptions.PreventUpdate

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()
    da_by_var = filter_data_request.compute()
    colors_by_var = charts.get_color_mapping(da_by_var)

    da_by_var = toolz.keyfilter(lambda v: v in vs, da_by_var)
    if len(da_by_var) == 0:
        raise dash.exceptions.PreventUpdate
        # return {
        #     'fig': charts.empty_figure(),
        #     'rng': [0, 1]
        # }

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)
    variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    yaxis_label_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    t_min, t_max = _get_min_max_time(da_by_var)

    series_by_var = toolz.valmap(lambda da: da.to_series(), da_by_var)

    # ORIGINAL TIME SERIES
    width = 1200
    height = 400
    orig_timeseries_fig = charts.multi_line(
        series_by_var,
        width=width, height=height,
        variable_label_by_var=variable_label_by_var,
        yaxis_label_by_var=yaxis_label_by_var,
        color_mapping=colors_by_var,
        subsampling=5_000,
    )

    # show title, legend, watermark, etc.
    orig_timeseries_fig.update_layout(
        legend=dict(orientation='h'),
        title='Original timeseries',
        xaxis={'title': 'time'},
        uirevision=integrate_datasets_request_hash,
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    orig_timeseries_fig = charts.add_watermark(orig_timeseries_fig)

    # if dash.ctx.triggered_id != FILTER_DATA_REQUEST_ID:
    #     # we reset the zoom only if a new filter data request was launched
    #     fig = charts.apply_figure_extent(fig, relayout_data)

    # print(f'get_plot_callback fig size={len(orig_timeseries_fig.to_json()) / 1e3}k')
    orig_timeseries_fig_data = {
        'fig': orig_timeseries_fig,
        'rng': [t_min, t_max],
    }

    # TREND
    if time_rng is not None:
        sel_variable, sel_t_min, sel_t_max = time_rng['variable_label'], time_rng['x_sel_min'], time_rng['x_sel_max']
        da_by_var = toolz.valmap(
            lambda da: da.sel({sel_variable: slice(sel_t_min, sel_t_max)}),
            da_by_var
        )
    else:
        sel_t_min, sel_t_max = None, None

    series_by_var = toolz.valmap(analysis._to_series, da_by_var)

    if do_aggregate:
        agg_func = trend_analysis_layout.AGGREGATION_FUNCTIONS[aggregation_function]
        get_aggregated_da_by_var = broadcast([0])(analysis.aggregate)
        series_by_var = get_aggregated_da_by_var(series_by_var, aggregation_period, agg_func, min_sample_size=min_sample_size)

    if do_deseasonize:
        series_by_var = toolz.valmap(
            lambda series: series - analysis.extract_seasonality(series),
            series_by_var
        )
        series_by_var = toolz.valmap(
            lambda series: analysis.moving_average(
                series,
                window=trend_analysis_layout.AGGREGATION_PERIOD_TIMEDELTA[moving_average_period]
            ),
            series_by_var
        )

    width = 1200
    height = 400
    trend_fig = charts.multi_line(
        series_by_var,
        width=width, height=height,
        variable_label_by_var=variable_label_by_var,
        yaxis_label_by_var=yaxis_label_by_var,
        color_mapping=colors_by_var,
        subsampling=5_000,
    )

    # show title, legend, watermark, etc.
    trend_fig.update_layout(
        legend=dict(orientation='h'),
        title='Trend',
        xaxis={'title': 'time'},
        # uirevision=integrate_datasets_request_hash,
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    trend_fig = charts.add_watermark(trend_fig)

    return (
        orig_timeseries_fig_data,
        trend_fig,
    )
