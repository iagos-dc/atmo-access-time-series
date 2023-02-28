import dash_bootstrap_components as dbc
from dash import dcc

from utils import dash_dynamic_components as ddc


DATA_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'data-analysis-variables-checklist-all-none-switch'
DATA_ANALYSIS_VARIABLES_CARDBODY_ROW_2_ID = 'data-analysis-variables-cardbody-row-2'
DATA_ANALYSIS_VARIABLES_CHECKLIST_ID = 'data-analysis-variables-checklist'


def _get_variables_checklist():
    select_all_none_variable_switch = dbc.Switch(
        id=ddc.add_active_to_component_id(DATA_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID),
        label='Select all / none',
        # style={'margin-top': '10px'},
        value=True,
    )

    variables_checklist = dbc.Card([
        dbc.CardHeader('Variables'),
        dbc.CardBody([
            dbc.Row(select_all_none_variable_switch),
            dbc.Row(id=ddc.add_active_to_component_id(DATA_ANALYSIS_VARIABLES_CARDBODY_ROW_2_ID)),
        ]),
    ])

    return variables_checklist


variables_checklist = _get_variables_checklist()
