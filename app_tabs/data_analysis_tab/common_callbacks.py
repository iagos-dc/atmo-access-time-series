import warnings
import dash
import dash_bootstrap_components as dbc
import toolz
from dash import Input

from . import common_layout
import data_processing
from app_tabs.common.layout import FILTER_DATA_REQUEST_ID
from data_processing import metadata
from log import log_exception
from utils import dash_dynamic_components as ddc, dash_persistence
from utils.exception_handler import callback_with_exc_handling, dynamic_callback_with_exc_handling, AppException, AppWarning


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(common_layout.DATA_ANALYSIS_VARIABLES_CARDBODY_ROW_2_ID, 'children'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(common_layout.DATA_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    prevent_initial_call=False,
)
@log_exception
def get_variables_callback(filter_data_request, variables_checklist_all_none_switch):
    def get_checklist(_id, options=None, value=None, **kwargs):
        if options is None:
            options = []
        return dbc.Checklist(id=_id, options=options, value=value, **kwargs)

    checklist_id = ddc.add_active_to_component_id(common_layout.DATA_ANALYSIS_VARIABLES_CHECKLIST_ID)

    if filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = filter_data_request.compute()
    da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)

    vs = list(metadata_by_var)
    if len(vs) == 0:
        warnings.warn('No variables found. Choose another dataset(s).', category=AppWarning)
        return get_checklist(_id=checklist_id)

    options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
    if variables_checklist_all_none_switch:
        value = vs
    else:
        value = []

    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()

    return get_checklist(
        _id=checklist_id,
        options=options,
        value=value,
        **dash_persistence.get_dash_persistence_kwargs(persistence_id=integrate_datasets_request_hash)
    )
