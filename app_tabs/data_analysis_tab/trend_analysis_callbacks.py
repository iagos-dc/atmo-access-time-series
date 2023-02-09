import numpy as np
import dash
import toolz
from dash import Input
import dash_bootstrap_components as dbc

import data_processing
from data_processing import metadata, analysis
from app_tabs.common import layout as common_layout
from app_tabs.data_analysis_tab import trend_analysis_layout
from log import log_exception, print_callback
from utils import dash_dynamic_components as ddc, charts, dash_persistence, helper
from utils.broadcast import broadcast


@ddc.dynamic_callback(
    ddc.DynamicOutput(trend_analysis_layout.TREND_GRAPH_ID, 'figure'),
    ddc.DynamicOutput(trend_analysis_layout.TREND_SUMMARY_CONTAINER_ID, 'children'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
)
@log_exception
@print_callback()
def get_trend_plot_callback(
        filter_data_request,
):
    if helper.any_is_None(filter_data_request):
        raise dash.exceptions.PreventUpdate

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()
    da_by_var = filter_data_request.compute()
    # colors_by_var = charts.get_color_mapping(da_by_var)

    if len(da_by_var) == 0:
        return charts.empty_figure(), None

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

    return charts.empty_figure(), children
