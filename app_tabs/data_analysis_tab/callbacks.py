import toolz
import numpy as np
import xarray as xr
import pandas as pd
import dash
from dash import Input, State, Output, callback
import plotly.express as px

from ..common import layout as common_layout
from . import layout
import data_processing
from data_processing import analysis
from utils import charts, combo_input_AIO
from log.log import log_exectime


@callback(
    Output(layout.VARIABLES_CHECKLIST_ID, 'options'),
    Output(layout.VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    Input(layout.VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    # prevent_initial_call=True,
)
def get_variables_callback(filter_data_request, variables_checklist_all_none_switch):
    trigger = dash.ctx.triggered_id
    if trigger is None or filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    vs = sorted(da_by_var)
    if len(vs) == 0:
        raise dash.exceptions.PreventUpdate

    options = [{'label': v, 'value': v} for v in vs]
    if variables_checklist_all_none_switch:
        value = vs
    else:
        value = []

    return options, value


@callback(
    Output(layout.ANALYSIS_METHOD_PARAMETERS_CARD_BODY_ID, 'children'),
    Output(layout.EXTRA_GRAPH_COMBO_INPUT_AIO_ID, 'children'),
    Input(layout.ANALYSIS_METHOD_RADIO_ID, 'value'),
)
def get_data_analysis_specification_store(analysis_method):
    if analysis_method == layout.GAUSSIAN_MEAN_AND_STD_METHOD:
        return (
            layout.gaussian_mean_and_std_parameters_combo_input,
            layout.gaussian_mean_and_std_extra_graph_controllers_combo_input
        )
    elif analysis_method == layout.PERCENTILES_METHOD:
        return (
            layout.percentiles_parameters_combo_input,
            layout.percentiles_extra_graph_controllers_combo_input,
        )


@callback(
    Output(layout.GRAPH_ID, 'figure'),
    Input(layout.VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    Input(combo_input_AIO.get_combo_input_data_store_id(layout.EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID), 'data'),
    Input(layout.GRAPH_SCATTER_MODE_RADIO_ID, 'value'),
    State(layout.ANALYSIS_METHOD_RADIO_ID, 'value'),
    State(layout.GRAPH_ID, 'relayoutData'),
    prevent_initial_call=True,
)
@log_exectime
def get_plot_callback(vs, filter_data_request, method_inputs, scatter_mode, analysis_method, relayout_data):
    print(f'relayout_data={relayout_data}')
    if any(map(
            lambda obj: obj is None,
            (dash.ctx.triggered_id, filter_data_request, analysis_method, method_inputs)
    )):
        raise dash.exceptions.PreventUpdate

    print(f'method_inputs={method_inputs}')

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    colors_by_var = charts.get_color_mapping(da_by_var)

    da_by_var = toolz.keyfilter(lambda v: v in vs, da_by_var)
    if len(da_by_var) == 0:
        return charts.empty_figure()

    if analysis_method == layout.GAUSSIAN_MEAN_AND_STD_METHOD:
        analysis_spec = method_inputs['value'][layout.GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID]
        aggregation_period = analysis_spec[layout.AGGREGATION_PERIOD_RADIO_ID]
        min_sample_size = analysis_spec[layout.MIN_SAMPLE_SIZE_INPUT_ID]

        mean, std, _ = analysis.gaussian_mean_and_std(da_by_var, aggregation_period, min_sample_size=min_sample_size)

        graph_controllers = method_inputs['value'][layout.EXTRA_GRAPH_COMBO_INPUT_AIO_ID]
        show_std = graph_controllers[layout.SHOW_STD_SWITCH_ID]
        std_mode = graph_controllers[layout.STD_MODE_RADIO_ID]

        width = 1200
        fig = charts.multi_line(
            mean,
            df_std=std if show_std else None,
            std_mode=std_mode,
            width=width, height=600,
            scatter_mode=scatter_mode,
            color_mapping=colors_by_var,
        )
    elif analysis_method == layout.PERCENTILES_METHOD:
        analysis_spec = method_inputs['value'][layout.PERCENTILES_COMBO_INPUT_AIO_ID]
        aggregation_period = analysis_spec[layout.AGGREGATION_PERIOD_RADIO_ID]
        min_sample_size = analysis_spec[layout.MIN_SAMPLE_SIZE_INPUT_ID]
        percentiles = analysis_spec[layout.PERCENTILES_CHECKLIST_ID]
        if analysis_spec[layout.PERCENTILE_USER_DEF_CHECKBOX_ID]:
            user_def_percentile = analysis_spec[layout.PERCENTILE_USER_DEF_INPUT_ID]
            if user_def_percentile is not None:
                percentiles = list(percentiles) + [user_def_percentile]

        quantiles, _ = analysis.percentiles(
            da_by_var,
            aggregation_period,
            p=sorted(percentiles),
            min_sample_size=min_sample_size
        )
        qs = np.sort(quantiles['quantile'].values)
        quantiles_df = {}
        for v, da in quantiles.data_vars.items():
            quantiles_df[v] = {}
            for q in qs:
                if q == 0:
                    p = 'min'
                elif q == 1:
                    p = 'max'
                else:
                    p = f'{int(100 * q)}'
                df = da.sel({'quantile': q}, drop=True).to_series()
                quantiles_df[v][p] = df

        width = 1200
        fig = charts.multi_line(
            quantiles_df,
            width=width, height=600,
            scatter_mode=scatter_mode,
            color_mapping=colors_by_var,
            line_dash_style_by_sublabel=layout.LINE_DASH_STYLE_BY_PERCENTILE,
        )
    else:
        raise dash.exceptions.PreventUpdate
        #raise NotImplementedError(analysis_method)

    # show title, legend, watermark, etc.
    fig.update_layout(
        legend=dict(orientation='h'),
        title=f'{analysis_method} for {", ".join(vs)}',
        hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    fig = charts.add_watermark(fig)

    print(f'get_plot_callback fig size={len(fig.to_json()) / 1e3}k')

    if dash.ctx.triggered_id != common_layout.FILTER_DATA_REQUEST_ID:
        # we reset the zoom only if a new filter data request was launched
        fig = charts.apply_figure_extent(fig, relayout_data)
    return fig
