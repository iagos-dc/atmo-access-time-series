import dash
from dash import Input, State, Output, callback
import plotly.express as px

from ..common.layout import FILTER_DATA_REQUEST_ID
from .layout import VARIABLES_CHECKLIST_ID, GRAPH_ID, RADIO_ID
import data_processing


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
    # prevent_initial_call=True,
)
def get_plot_callback(vs, filter_data_request, param):
    if dash.ctx.triggered_id is None or filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    fig_by_var = {}
    for v, da in da_by_var.items():
        if v in vs:
            da_mean = da.resample({'time': param}).mean()
            df = da_mean.to_dataframe()
            fig_by_var[v] = px.line(df, y=v)

    if len(fig_by_var) == 0:
        raise dash.exceptions.PreventUpdate

    return list(fig_by_var.values())[0]
