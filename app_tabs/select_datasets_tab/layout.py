import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from utils.dash_persistence import get_dash_persistence_kwargs
from app_tabs.common.layout import SELECT_DATASETS_TAB_VALUE, NON_INTERACTIVE_GRAPH_CONFIG, get_next_button, std_variables

VARIABLES_LEGEND_DROPDOWN_ID = 'variables-legend-dropdown'
# 'options' contains a list of dictionaries {'label' -> variable label, 'value' -> variable description}
# 'value' contains a list of variable labels in the dropdown

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
    gantt_view_radio = dbc.InputGroup(
        [
            dbc.InputGroupText('View: ', style={'margin-right': '10px'}),
            dbc.RadioItems(
                id=GANTT_VIEW_RADIO_ID,
                options=[
                    {'label': 'compact', 'value': 'compact'},
                    {'label': 'detailed', 'value': 'detailed'},
                ],
                value='compact',
                inline=True
            ),
        ],
        size='lg',
        style={
            'display': 'flex',
            'align-items': 'center',
            'border': '1px solid lightgrey',
            'border-radius': '5px'
        }
    )

    reset_gantt_selection_button = dbc.Button(
        id=RESET_DATASETS_SELECTION_BUTTON_ID,
        n_clicks=0,
        outline=True,
        color='secondary',
        type='submit',
        style={'font-weight': 'bold'},
        size='lg',
        children='Clear',
    )

    gantt_graph = dcc.Graph(
        id=GANTT_GRAPH_ID,
        config=NON_INTERACTIVE_GRAPH_CONFIG,
        # style={'height': '100%'},
    )

    all_none_switch = dbc.Switch(
        id=DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID,
        label='Select all / none',
        style={'margin-top': '10px'},
        value=False,
    )

    table_col_ids = ['eye', 'title', 'var_codes_filtered', 'RI', 'long_name', 'platform_id', 'time_period_start', 'time_period_end',
                     #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                     ]
    table_col_names = ['Plot', 'Title', 'Variables', 'RI', 'Station', 'Station code', 'Start', 'End',
                       #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                       ]
    table_columns = [{'name': name, 'id': i} for name, i in zip(table_col_names, table_col_ids)]
    # on rendering HTML snipplets in DataTable cells:
    # https://github.com/plotly/dash-table/pull/916
    table_columns[0]['presentation'] = 'markdown'

    table = dash_table.DataTable(
        id=DATASETS_TABLE_ID,
        columns=table_columns,
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
        style_cell_conditional=[
            {
                'if': {'column_id': 'eye'},
                'textAlign': 'center',
            }
        ],
        style_header={'fontWeight': 'bold'},
        markdown_options={'html': True},
    )

    quicklook_popup = html.Div(id=QUICKLOOK_POPUP_ID)

    gantt_diagram_card = dbc.Card(
        [
            dbc.CardHeader(
                html.Div([
                    html.B('a) Datasets time coverage:'), ' ', 'Click on stations to see the datasets in the table on the right'
                ]),
            ),
            dbc.CardBody(
                [
                    dbc.Row(dbc.Col(gantt_graph)),
                ],
                style={'overflowY': 'scroll'}
            ),
            dbc.CardFooter(
                html.Div(
                    [
                        html.Div(reset_gantt_selection_button),
                        html.Div(gantt_view_radio),
                    ],
                    style={
                        'display': 'flex',
                        'justify-content': 'space-between',
                        'align-items': 'center',
                    },
                ),
            )
        ],
        style={'height': '70vh'}  # 70% of viewport height
    )

    datasets_table_card = dbc.Card([
        dbc.CardHeader(
            'b) Select your datasets here',
            style={'font-weight': 'bold'},
        ),
        dbc.CardBody([
            dbc.Row(dbc.Col(all_none_switch)),
            dbc.Row(dbc.Col(table)),
        ])
    ])

    return dbc.Tab(
        label='2. Select datasets',
        id=SELECT_DATASETS_TAB_VALUE,
        tab_id=SELECT_DATASETS_TAB_VALUE,
        disabled=True,

        children=[
            html.Div(
                style={'margin-top': '5px', 'margin-left': '20px', 'margin-right': '20px'},
                children=[
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.B('Variables legend:'),
                                    dcc.Dropdown(
                                        id=VARIABLES_LEGEND_DROPDOWN_ID,
                                        options=std_variables.to_dict(orient='records'),
                                        multi=True,
                                        clearable=False,
                                        disabled=True,
                                    ),
                                ],
                                width=10
                            ),
                            dbc.Col(
                                children=html.Div(get_next_button(SELECT_DATASETS_BUTTON_ID), style={'display': 'flex', 'justify-content': 'end'}),
                                width=2,
                            ),
                        ],
                        justify='between',
                        style={'margin-bottom': '10px'},
                    ),
                    dbc.Row([
                        dbc.Col(
                            width=5,
                            #style={'height': '100%'},
                            children=gantt_diagram_card
                        ),
                        dbc.Col(
                            width=7,
                            children=datasets_table_card
                        )
                    ])
                ]
            ),
            quicklook_popup
        ]
    )
