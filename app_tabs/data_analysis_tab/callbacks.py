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
from utils import charts


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
    Output(layout.DATA_ANALYSIS_SPECIFICATION_STORE_ID, 'data'),
    Input(layout.ANALYSIS_METHOD_RADIO_ID, 'value'),
    Input(layout.RADIO_ID, 'value'),
    Input(layout.SHOW_STD_SWITCH_ID, 'value')
)
def get_data_analysis_specification_store(analysis_method, params, show_std):
    analysis_spec = {
        'method': analysis_method,
        'aggregation_period': params,
        'show_std': show_std,
    }
    print(analysis_spec)
    return analysis_spec


@callback(
    Output(layout.GRAPH_ID, 'figure'),
    Input(layout.VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    Input(layout.DATA_ANALYSIS_SPECIFICATION_STORE_ID, 'data'),
    prevent_initial_call=True,
)
def get_plot_callback(vs, filter_data_request, analysis_spec):
    if dash.ctx.triggered_id is None or filter_data_request is None or analysis_spec is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    colors_by_var = charts.get_color_mapping(da_by_var)

    da_by_var = toolz.keyfilter(lambda v: v in vs, da_by_var)
    if len(da_by_var) == 0:
        return charts.empty_figure()

    analysis_method = analysis_spec['method']
    if analysis_method == layout.GAUSSIAN_MEAN_AND_STD_METHOD:
        aggregation_period = analysis_spec['aggregation_period']
        show_std = analysis_spec['show_std']
        mean, std, _ = analysis.gaussian_mean_and_std(da_by_var, aggregation_period, calc_std=show_std)
        width = 1200
        fig = charts.multi_line(
            mean.to_dataframe(),
            df_std=std.to_dataframe() if std is not None else None,
            width=width, height=600, color_mapping=colors_by_var
        )
    else:
        raise NotImplementedError(analysis_method)

    # show title, legend, watermark, etc.
    fig.update_layout(
        legend=dict(orientation='h'),
        title=f'{analysis_method} for {", ".join(vs)}',
    )
    fig = charts.add_watermark(fig)
    return fig
