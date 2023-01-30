from dash import Input, callback, Output

from app_tabs.data_analysis_tab.layout import DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, DATA_ANALYSIS_FIGURE_CONTAINER_ID, \
    KIND_OF_ANALYSIS_TABS_ID, EXPLORATORY_ANALYSIS_TAB_ID, TREND_ANALYSIS_TAB_ID, MULTIVARIATE_ANALYSIS_TAB_ID
from app_tabs.data_analysis_tab.exploratory_analysis_layout import exploratory_analysis_cardbody, \
    exploratory_plot
from app_tabs.data_analysis_tab.multivariate_analysis_layout import multivariate_analysis_cardbody, multivariate_plot
from log import log_exception


# @ddc.dynamic_callback(
#     ddc.DynamicOutput(layout.X_VARIABLE_SELECT_ID, 'options'),
#     ddc.DynamicOutput(layout.X_VARIABLE_SELECT_ID, 'value'),
#     ddc.DynamicOutput(layout.Y_VARIABLE_SELECT_ID, 'options'),
#     ddc.DynamicOutput(layout.Y_VARIABLE_SELECT_ID, 'value'),
#     ddc.DynamicOutput(layout.C_VARIABLE_SELECT_ID, 'options'),
#     ddc.DynamicOutput(layout.C_VARIABLE_SELECT_ID, 'value'),
#     Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
#     prevent_initial_call=False,
# )
# @log_exception
# def get_x_y_c_variables_callback(filter_data_request):
#     if filter_data_request is None:
#         raise dash.exceptions.PreventUpdate
#
#     req = data_processing.FilterDataRequest.from_dict(filter_data_request)
#     da_by_var = req.compute()
#     da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
#     metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)
#
#     vs = list(metadata_by_var)
#     if len(vs) <= 1:
#         return (None, ) * 6
#
#     options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
#     options_c = [{'label': '---', 'value': '---'}]
#     if len(vs) >= 3:
#         options_c.extend(options)
#
#     return options, vs[0], options, vs[1], options_c, '---',


@callback(
    Output(DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, 'children'),
    Output(DATA_ANALYSIS_FIGURE_CONTAINER_ID, 'children'),
    Input(KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
)
@log_exception
def get_data_analysis_carbody_content(tab_id):
    if tab_id == EXPLORATORY_ANALYSIS_TAB_ID:
        param_cardbody_children = exploratory_analysis_cardbody
        figure_container_children = exploratory_plot
    elif tab_id == TREND_ANALYSIS_TAB_ID:
        param_cardbody_children = []
        figure_container_children = []
    elif tab_id == MULTIVARIATE_ANALYSIS_TAB_ID:
        param_cardbody_children = multivariate_analysis_cardbody
        figure_container_children = multivariate_plot
    else:
        raise ValueError(f'unknown tab_id={tab_id}')
    return param_cardbody_children, figure_container_children
