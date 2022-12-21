import toolz
import numpy as np
import xarray as xr
import pandas as pd
import dash
from dash import Input, State, Output, callback
import plotly.express as px

from ..common.layout import FILTER_DATA_REQUEST_ID
from .layout import VARIABLES_CHECKLIST_ID, GRAPH_ID, RADIO_ID
import data_processing
from utils import charts


@callback(
    Output(VARIABLES_CHECKLIST_ID, 'options'),
    Output(VARIABLES_CHECKLIST_ID, 'value'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
    # prevent_initial_call=True,
)
def get_variables_callback(filter_data_request):
    if dash.ctx.triggered_id is None or filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    vs = sorted(da_by_var)
    if len(vs) == 0:
        raise dash.exceptions.PreventUpdate

    options = [{'label': v, 'value': v} for v in vs]
    value = [vs[0]]
    return options, value


@callback(
    Output(GRAPH_ID, 'figure'),
    Input(VARIABLES_CHECKLIST_ID, 'value'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
    Input(RADIO_ID, 'value'),
    prevent_initial_call=True,
)
def get_plot_callback(vs, filter_data_request, param):
    if dash.ctx.triggered_id is None or filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    da_by_var_filtered = toolz.keyfilter(lambda v: v in vs, da_by_var)
    if len(da_by_var_filtered) == 0:
        return charts.empty_figure()

    da_agg_by_var_filtered = toolz.valmap(lambda da: da.resample({'time': param}).mean(), da_by_var_filtered)
    df = xr.Dataset(da_agg_by_var_filtered).reset_coords(drop=True).to_dataframe()
    colors_by_var = charts.get_color_mapping(da_by_var)

    width = 1200
    fig = charts.multi_line(df, width=width, height=600, color_mapping=colors_by_var)

    # show title, legend, watermark, etc.
    nvars = len(da_by_var_filtered)
    delta_domain = min(75 / width, 0.5 / nvars)
    annotations = [dict(
        name="sdfs watermark",
        text="ATMO-ACCESS",
        textangle=-30,
        opacity=0.1,
        font=dict(color="black", size=75),
        xref="paper",
        yref="paper",
        x=0.5 - delta_domain * (nvars % 2),
        y=0.5,
        showarrow=False,
    )]
    fig.update_layout(
        legend=dict(orientation='h'),
        title='Here comes a title...',
        annotations=annotations,
    )

    return fig
