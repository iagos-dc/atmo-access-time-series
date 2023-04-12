import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from app_tabs.common.layout import SELECT_DATASETS_TAB_VALUE, NON_INTERACTIVE_GRAPH_CONFIG


GANTT_VIEW_RADIO_ID = 'gantt-view-radio'
# 'value' contains 'compact' or 'detailed'

GANTT_GRAPH_ID = 'gantt-graph'
# 'figure' contains a Plotly figure object

DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID = 'datasets-table-checklist-all-none-switch'

DATASETS_TABLE_ID = 'datasets-table'
# 'columns' contains list of dictionaries {'name' -> column name, 'id' -> column id}
# 'data' contains a list of records as provided by the method pd.DataFrame.to_dict(orient='records')

QUICKLOOK_POPUP_ID = 'quicklook-popup'
# 'children' contains a layout of the popup

SELECT_DATASETS_BUTTON_ID = 'select-datasets-button'
# 'n_click' contains a number of clicks at the button

RESET_DATASETS_SELECTION_BUTTON_ID = 'reset-datasets-selection-button'

BAR_UNSELECTED = 0
BAR_PARTIALLY_SELECTED = 1
BAR_SELECTED = 2

UNSELECTED_GANTT_OPACITY = 0.15
PARTIALLY_SELECTED_GANTT_OPACITY = 0.5
SELECTED_GANTT_OPACITY = 1.0

OPACITY_BY_BAR_SELECTION_STATUS = {
    BAR_UNSELECTED: UNSELECTED_GANTT_OPACITY,
    BAR_PARTIALLY_SELECTED: PARTIALLY_SELECTED_GANTT_OPACITY,
    BAR_SELECTED: SELECTED_GANTT_OPACITY,
}


def get_select_datasets_tab():
    gantt_view_radio = dbc.RadioItems(
        id=GANTT_VIEW_RADIO_ID,
        options=[
            {'label': 'compact view', 'value': 'compact'},
            {'label': 'detailed view', 'value': 'detailed'},
        ],
        value='compact',
        inline=True
    )

    select_datasets_button = dbc.Button(
        id=SELECT_DATASETS_BUTTON_ID,
        n_clicks=0,
        color='primary', type='submit',
        style={'font-weight': 'bold'},
        children='Select datasets'
    )

    reset_gantt_selection_button = dbc.Button(
        id=RESET_DATASETS_SELECTION_BUTTON_ID,
        n_clicks=0,
        color='primary',
        type='submit',
        style={'font-weight': 'bold'},
        children='Clear selection on the Gantt diagram',
    )

    gantt_graph = dcc.Graph(
        id=GANTT_GRAPH_ID,
        config=NON_INTERACTIVE_GRAPH_CONFIG,
    )

    all_none_switch = dbc.Switch(
        id=DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID,
        label='Select all / none',
        style={'margin-top': '10px'},
        value=False,
    )

    table = dash_table.DataTable(
        id=DATASETS_TABLE_ID,
        css=[dict(selector="p", rule="margin: 0px;")],
        # see: https://dash.plotly.com/datatable/interactivity
        row_selectable="multi",
        selected_rows=[],
        selected_row_ids=[],
        sort_action='native',
        # filter_action='native',
        page_action="native", page_current=0, page_size=20,
        # see: https://dash.plotly.com/datatable/width
        # hidden_columns=['url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
        },
        style_cell={'textAlign': 'left'},
        markdown_options={'html': True},
    )

    quicklook_popup = html.Div(id=QUICKLOOK_POPUP_ID)


    return dcc.Tab(
        label='Select datasets',
        id=SELECT_DATASETS_TAB_VALUE,
        value=SELECT_DATASETS_TAB_VALUE,
        disabled=True,
        children=[
            html.Div(
                style={'margin': '20px'},
                children=[
                    html.Div(id='select-datasets-left-panel-div', className='five columns', children=[
                        html.Div(id='select-datasets-1st-subpanel-div', className='twelve columns', children=select_datasets_button, style={'text-align': 'right', 'margin-bottom': '20px'}),
                        html.Div(id='select-datasets-2nd-subpanel-div', className='twelve columns', style={'margin-bottom': '10px'}, children=[
                            html.Div(id='select-datasets-2nd-left-subpanel-div', className='six columns', children=gantt_view_radio),
                            html.Div(id='select-datasets-2nd-right-subpanel-div', className='six columns', children=reset_gantt_selection_button, style={'text-align': 'right'}),
                        ]),
                        html.Div(id='select-datasets-3rd-subpanel-div2', className='twelve columns', children=gantt_graph),
                    ]),
                    html.Div(id='select-datasets-right-panel-div', className='seven columns', children=[
                        all_none_switch,
                        table,
                    ]),
    #                html.Div(id='select-datasets-main-panel-div', className='twelve columns', children=[
    #                ]),
                ]
            ),
            quicklook_popup,
        ]
    )
