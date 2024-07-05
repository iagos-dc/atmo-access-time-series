import warnings
import numpy as np
import pandas as pd
import toolz
import plotly
import plotly.graph_objects as go
import plotly.subplots
import dash
from dash import Input, Output, ALL
import dash_bootstrap_components as dbc

import data_processing
from data_processing import metadata, analysis
from app_tabs.common.layout import FILTER_DATA_REQUEST_ID
from . import common_layout, tabs_layout
from app_tabs.data_analysis_tab import trend_analysis_layout
from log import log_exception, log_profiler_info, dump_exception_to_log
from utils import dash_dynamic_components as ddc, charts, dash_persistence, helper
from utils.graph_with_horizontal_selection_AIO import figure_data_store_id, selected_range_store_id
from utils.exception_handler import dynamic_callback_with_exc_handling, AppWarning, AppException


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


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(trend_analysis_layout.AGGREGATE_COLLAPSE_ID, 'is_open'),
    Input(tabs_layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATE_CHECKBOX_ID, 'value'),
    prevent_initial_call=True
)
@log_exception
def show_aggregate_card(tab_id, aggregate_checkbox):
    if tab_id != tabs_layout.TREND_ANALYSIS_TAB_ID:
        raise dash.exceptions.PreventUpdate
    return aggregate_checkbox


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(trend_analysis_layout.APPLY_MOVING_AVERAGE_CHECKBOX_ID, 'disabled'),
    ddc.DynamicOutput(trend_analysis_layout.APPLY_MOVING_AVERAGE_COLLAPSE_ID, 'is_open'),
    ddc.DynamicOutput(trend_analysis_layout.MOVING_AVERAGE_PERIOD_SELECT_ID, 'disabled'),
    Input(tabs_layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
    ddc.DynamicInput(trend_analysis_layout.DESEASONIZE_CHECKBOX_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.APPLY_MOVING_AVERAGE_CHECKBOX_ID, 'value'),
    prevent_initial_call=True
)
@log_exception
def show_moving_average_card(tab_id, deseasonize_checkbox, apply_moving_average_checkbox):
    if tab_id != tabs_layout.TREND_ANALYSIS_TAB_ID:
        raise dash.exceptions.PreventUpdate
    return not deseasonize_checkbox, apply_moving_average_checkbox, not deseasonize_checkbox


#@log_profiler_info()
def _get_theil_sen_slope(series):
    (a, b), (ci0, ci1), (x_unit, y_unit) = data_processing.analysis.theil_sen_slope(series, subsampling=3000)
    return (a, b), (ci0, ci1), (x_unit, y_unit)


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(figure_data_store_id(trend_analysis_layout.TREND_ANALYSIS_AIO_ID + '-time', trend_analysis_layout.TREND_ANALYSIS_AIO_CLASS), 'data'),
    ddc.DynamicOutput(trend_analysis_layout.TREND_GRAPH_ID, 'figure'),
    ddc.DynamicOutput(trend_analysis_layout.AUTOCORRELATION_GRAPH_ID, 'figure'),
    ddc.DynamicOutput(trend_analysis_layout.TREND_SUMMARY_BAR_GRAPH_ID, 'figure'),
    Input(tabs_layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(common_layout.DATA_ANALYSIS_VARIABLES_CHECKLIST_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.TREND_ANALYSIS_METHOD_RADIO_ID, 'value'),
    ddc.DynamicInput(selected_range_store_id(trend_analysis_layout.TREND_ANALYSIS_AIO_ID + '-time', trend_analysis_layout.TREND_ANALYSIS_AIO_CLASS), 'data'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATE_CHECKBOX_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATION_PERIOD_SELECT_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.AGGREGATION_FUNCTION_SELECT_ID, 'value'),
    ddc.DynamicInput(common_layout.MIN_SAMPLE_SIZE_INPUT_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.DESEASONIZE_CHECKBOX_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.APPLY_MOVING_AVERAGE_CHECKBOX_ID, 'value'),
    ddc.DynamicInput(trend_analysis_layout.MOVING_AVERAGE_PERIOD_SELECT_ID, 'value'),
    prevent_initial_call=True
)
@log_exception
#@log_profiler_info()
def get_trend_plots_callback(
        tab_id,
        filter_data_request_as_dict,
        vs,
        analysis_method,
        time_rng,
        do_aggregate,
        aggregation_period,
        aggregation_function,
        min_sample_size,
        do_deseasonize,
        apply_moving_average,
        moving_average_period
):
    if tab_id != tabs_layout.TREND_ANALYSIS_TAB_ID:
        raise dash.exceptions.PreventUpdate

    args = (
        tab_id, filter_data_request_as_dict, vs, analysis_method, time_rng, do_aggregate, aggregation_period,
        aggregation_function, min_sample_size, do_deseasonize, apply_moving_average,
        moving_average_period
    )
    if helper.any_is_None(
        filter_data_request_as_dict,
        vs,
        analysis_method,
        do_aggregate, aggregation_period, aggregation_function, min_sample_size,
        do_deseasonize, apply_moving_average, moving_average_period
    ):
        raise dash.exceptions.PreventUpdate

    # print(f'get_trend_plots_callback with ctx={list(dash.ctx.triggered_prop_ids.values())}')

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request_as_dict)
    integrate_datasets_request = filter_data_request.integrate_datasets_request
    colors_by_var = charts.get_color_mapping(integrate_datasets_request.compute())
    integrate_datasets_request_hash = integrate_datasets_request.deterministic_hash()
    da_by_var = filter_data_request.compute()

    da_by_var = toolz.keyfilter(lambda v: v in vs, da_by_var)
    colors_by_var = toolz.keyfilter(lambda v: v in vs, colors_by_var)
    if len(da_by_var) == 0:
        warnings.warn('No variable selected. Please select one.', category=AppWarning)
        empty_fig_data = {
            'fig': charts.empty_figure(),
            'rng': [None, None],
        }
        return (empty_fig_data, ) + (charts.empty_figure(), ) * 3

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)
    # variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    yaxis_label_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)
    variable_id_by_var = dict(zip(list(metadata_by_var), list(metadata_by_var)))

    t_min, t_max = _get_min_max_time(da_by_var)

    series_by_var = toolz.valmap(analysis._to_series, da_by_var)

    # ORIGINAL TIME SERIES
    height = 300
    orig_timeseries_fig = charts.multi_line(
        series_by_var,
        height=height,
        variable_label_by_var=variable_id_by_var,
        # variable_label_by_var=variable_label_by_var,
        yaxis_label_by_var=yaxis_label_by_var,
        color_mapping=colors_by_var,
        subsampling=10_000,
    )

    # show title, legend, watermark, etc.
    orig_timeseries_fig.update_layout(
        autosize=True,
        margin={'t': 0, 'l': 15, 'r': 15},
        # legend={
        #     'xanchor': 'right',
        #     'yanchor': 'top',
        #     'x': 0.99,
        #     'y': 0.99,
        # },
        # legend_y=-0.4,
        # legend_y=-0.4,
        # title='Original timeseries (setup time filter here)',
        title='',
        # xaxis={'title': 'time'},
        xaxis={'title': ''},
        uirevision=integrate_datasets_request_hash,
        hovermode=False  # turn-off the hover
    )
    orig_timeseries_fig_data = {
        'fig': orig_timeseries_fig,
        'rng': [t_min, t_max],
    }

    # TREND CALCULATION
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
        aggregated_series_by_var = {}
        for v, series in series_by_var.items():
            aggregated_series = analysis.aggregate(series, aggregation_period, agg_func, min_sample_size=min_sample_size)
            if aggregated_series.isna().all():
                warnings.warn(
                    f'Not enough data to calculate the moving average for {v}. '
                    f'Try a larger window size, decrease the minimal sample size per window or change the data filter.',
                    category=AppWarning
                )
                continue
            aggregated_series_by_var[v] = aggregated_series
        series_by_var = aggregated_series_by_var

    if do_deseasonize:
        deseasonized_series_by_var = {}
        for v, series in series_by_var.items():
            try:
                deseasonized_series = series - analysis.extract_seasonality(series)
            except AssertionError as e:
                dump_exception_to_log(e, func=get_trend_plots_callback, args=args)
                warnings.warn(
                    f'Cannot estimate a seasonal component for the variable {v}. '
                    f'Possible cause: not enought data or time span is less 2 years. '
                    f'Please try to change the filtering criteria or deselect this variable.',
                    category=AppWarning
                )
                continue
            deseasonized_series_by_var[v] = deseasonized_series
        series_by_var = deseasonized_series_by_var

    if do_deseasonize and apply_moving_average:
        series_by_var = toolz.valmap(
            lambda series: analysis.moving_average(
                series,
                window=trend_analysis_layout.AGGREGATION_PERIOD_TIMEDELTA[moving_average_period]
            ),
            series_by_var
        )

    theil_sen_slope_by_var = {}
    for v, series in series_by_var.items():
        try:
            theil_sen_slope_by_var[v] = _get_theil_sen_slope(series)
        except ValueError as e:
            dump_exception_to_log(e, func=get_trend_plots_callback, args=args)
            warnings.warn(
                f'Cannot estimate trend for the variable {v}. '
                f'Possible cause: not enough data available. '
                f'Please try to change the filtering criteria or deselect this variable.',
                category=AppWarning
            )

    if do_deseasonize:
        TIMESERIES_LEGEND_SUBLABEL = 'time series without seasonal component'
    else:
        TIMESERIES_LEGEND_SUBLABEL = 'time series'
    if do_aggregate:
        TIMESERIES_LEGEND_SUBLABEL = f'aggregated {TIMESERIES_LEGEND_SUBLABEL}'

    series_and_trend_line_by_var = {}
    stationary_series_by_var = {}
    trend_by_var = {}
    trend_error_by_var = {}
    trend_unit_by_var = {}
    for v in theil_sen_slope_by_var:
        series = series_by_var[v]
        (a, b), (ci0, ci1), (x_unit, y_unit) = theil_sen_slope_by_var[v]
        t = series.index #.to_numpy(dtype="M8[ns]")
        # t = np.array([np.datetime64(series.index[i]) for i in (0, -1)])
        t_in_seconds = (t - analysis.BASE_DATE) / np.timedelta64(1, 's')
        x = a * t_in_seconds + b
        trend_series = pd.Series(x, index=t)
        stationary_series_by_var[v] = series - trend_series
        series_and_trend_line_by_var[v] = {
            TIMESERIES_LEGEND_SUBLABEL: series,
            'trend': trend_series.iloc[[0, -1]]
        }

        a_in_y_units_per_year = a * y_unit * (np.timedelta64(365 * 24 * 3600, 's') / x_unit)
        ci0_in_y_units_per_year = ci0 * y_unit * (np.timedelta64(365 * 24 * 3600, 's') / x_unit)
        ci1_in_y_units_per_year = ci1 * y_unit * (np.timedelta64(365 * 24 * 3600, 's') / x_unit)
        md = metadata_by_var.get(v, {})
        units = md.get(metadata.UNITS, '???')
        # trend_summary = f'{v} : {a:.4g} {units} / year; 95% CI: [{ci0:.4g}, {ci1:.4g}] {units} / year'
        trend_by_var[v] = a_in_y_units_per_year
        trend_error_by_var[v] = (ci1_in_y_units_per_year - ci0_in_y_units_per_year) / 2
        trend_unit_by_var[v] = f'{units} / year'

    trend_summary_df = pd.DataFrame.from_dict({
        'trend': trend_by_var,
        'trend_error': trend_error_by_var,
        'trend_unit': trend_unit_by_var,
        'color': toolz.valmap(plotly.colors.label_rgb, colors_by_var)
    })

    autocorr_by_var = toolz.valmap(analysis.autocorrelation, stationary_series_by_var)

    # TREND FIGURE
    # width = 'auto' #1200
    height = 500
    trend_fig = charts.multi_line(
        series_and_trend_line_by_var,
        # width=width,
        height=height,
        variable_label_by_var=variable_id_by_var,
        # variable_label_by_var=variable_label_by_var,
        yaxis_label_by_var=yaxis_label_by_var,
        color_mapping=colors_by_var,
        line_dash_style_by_sublabel={
            TIMESERIES_LEGEND_SUBLABEL: 'solid',
            'trend': 'dash',
        },
        marker_opacity_by_sublabel={
            TIMESERIES_LEGEND_SUBLABEL: 0.4,
            'trend': 1,
        },
        subsampling=5_000,
    )

    # show title, legend, watermark, etc.
    trend_fig.update_layout(
        autosize=True,
        margin={'l': 5, 'r': 5},
        legend={
            'xanchor': 'left',
            'yanchor': 'top',
            'x': 0,
            'y': -0.3,
        },
        #legend_y=-0.4,
        title='Trend',
        xaxis={'title': 'time'},
        # uirevision=integrate_datasets_request_hash,
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
        hovermode='x unified'  # https://plotly.com/python/hover-text-and-formatting/#control-hovermode-with-dash
    )
    trend_fig = charts.add_watermark(trend_fig)

    # AUTOCORR FIGURE
    autocorr_fig = go.Figure()
    for variable_id, autocorr in autocorr_by_var.items():
        autocorr_fig_trace = charts.plotly_scatter(
            x=autocorr.index.values,
            y=autocorr.values,
            legendgroup=variable_id,
            marker_color=plotly.colors.label_rgb(colors_by_var[variable_id]),
            name=variable_id,
            # name=variable_label_by_var[variable_id],
        )
        autocorr_fig.add_trace(autocorr_fig_trace)

    # show title, legend, watermark, etc.
    autocorr_fig.update_layout(
        autosize=True,
        margin={'l': 5, 'r': 5},
        height=400,
        legend={
            'orientation': 'h',
            'xanchor': 'right',
            'yanchor': 'top',
            'x': 0.99,
            'y': 0.99,
        },
        title='Autocorrelation',
        xaxis={
            'title': 'lag (days)',
            'zeroline': True,
            'zerolinecolor': 'grey',
            'zerolinewidth': 1,
        },
        yaxis={
            'title': 'correlation',
            'zeroline': True,
            'zerolinecolor': 'grey',
            'zerolinewidth': 1,
        },
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
        hovermode='x unified'  # https://plotly.com/python/hover-text-and-formatting/#control-hovermode-with-dash
    )
    autocorr_fig = charts.add_watermark(autocorr_fig)

    # TREND SUMMARY ON BAR PLOT
    if len(trend_by_var) > 0:
        bars = []
        _series_by_unit = {}
        _error_series_by_unit = {}
        for units, trend_df_for_units in trend_summary_df.groupby('trend_unit'):
            _series_by_unit[units] = trend_df_for_units['trend']
            _error_series_by_unit[units] = trend_df_for_units['trend_error']

            bars_for_units = go.Bar(
                x=trend_df_for_units.index,
                y=trend_df_for_units['trend'],
                error_y={
                    'type': 'data',
                    'array': trend_df_for_units['trend_error'],
                },
                customdata=trend_df_for_units['trend_unit'],
                hovertemplate='%{y:.3g} &plusmn; %{error_y.array:.3g} %{customdata}<extra></extra>',
                marker_color=trend_df_for_units['color'],
                orientation='v',
            )
            bars.append(bars_for_units)

        _bars_count_by_unit = toolz.valmap(len, _series_by_unit)
        _bars_count_list = list(_bars_count_by_unit.values())
        single_bar_fraction_gap = 1.
        delta_domain = 1 / (sum(_bars_count_list) + (len(_bars_count_list) - 1) * single_bar_fraction_gap)
        domain_left_ends = np.hstack(([0], np.array(_bars_count_list[:-1]) + single_bar_fraction_gap)).cumsum() * delta_domain
        domain_right_ends = ((np.array(_bars_count_list) + single_bar_fraction_gap).cumsum() - single_bar_fraction_gap) * delta_domain

        range_tick0_dtick_by_unit = charts.get_range_tick0_dtick_by_var(_series_by_unit, nticks=5, error_df=_error_series_by_unit, align_zero=True)
        yaxis_props_by_unit = charts.get_sync_axis_props(range_tick0_dtick_by_unit)
        yaxes_props = []
        for units, yaxis_props in yaxis_props_by_unit.items():
            yaxis_props['title_text'] = units
            yaxis_props['title_standoff'] = 0
            yaxes_props.append(yaxis_props)

        trend_summary_fig = plotly.subplots.make_subplots(rows=1, cols=len(bars))
        xaxis_props_by_xaxis_id = {}
        yaxis_props_by_yaxis_id = {}
        for i, (bars_for_units, yaxis_props) in enumerate(zip(bars, yaxes_props)):
            trend_summary_fig.add_trace(bars_for_units, row=1, col=i + 1)
            xaxis_id = 'xaxis' if i == 0 else f'xaxis{i + 1}'
            xaxis_props_by_xaxis_id[xaxis_id] = {
                'fixedrange': True,
                'domain': [domain_left_ends[i], domain_right_ends[i]],
                'title': '',
            }
            yaxis_id = 'yaxis' if i == 0 else f'yaxis{i + 1}'
            yaxis_props_by_yaxis_id[yaxis_id] = yaxis_props
            yaxis_props_by_yaxis_id[yaxis_id].update({
                'anchor': 'free',
                'position': domain_left_ends[i],
                'side': 'left',
            })

        trend_summary_fig.update_layout(**xaxis_props_by_xaxis_id, **yaxis_props_by_yaxis_id)
    else:
        trend_summary_fig = charts.empty_figure(height=400)

    trend_summary_fig.update_layout(
        autosize=True,
        margin={'l': 5, 'r': 5},
        title='Trend rate and 95% CI',
        showlegend=False,
        height=400,
    )
    trend_summary_fig = charts.add_watermark(trend_summary_fig)

    return (
        orig_timeseries_fig_data,
        trend_fig,
        autocorr_fig,
        trend_summary_fig,
    )
