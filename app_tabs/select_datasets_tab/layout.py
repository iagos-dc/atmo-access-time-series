import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

import data_access
from app_tabs.common.layout import SELECT_DATASETS_TAB_VALUE, NON_INTERACTIVE_GRAPH_CONFIG, get_help_icon, get_next_button
from utils import colors

VARIABLES_LEGEND_DROPDOWN_ID = 'variables-legend-dropdown'
# 'options' contains a list of dictionaries {'label' -> variable label, 'value' -> variable description}
# 'value' contains a list of variable labels in the dropdown

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


def get_variables_legend_options(selected_ecvs):
    selected_ecvs = set(selected_ecvs)
    all_ECV = data_access.var_codes_by_ECV.index
    ECV_filter = all_ECV.isin(set(selected_ecvs))
    filtered_var_codes_by_ECV = data_access.var_codes_by_ECV[ECV_filter]
    color_by_ECV_name_dict = colors.get_color_by_ECV_name_dict()

    variables_legend_options = []
    for ecv, var_code in filtered_var_codes_by_ECV.to_dict().items():
        label = f'{var_code} - {ecv}'
        _color = color_by_ECV_name_dict[ecv]
        variables_legend_option = {
            'value': ecv,
            'label': html.P([html.I(className='fa-solid fa-square', style={'color': _color}), ' ', label], style={'font-size': '120%'})
        }
        variables_legend_options.append(variables_legend_option)
    return variables_legend_options


def get_select_datasets_tab():
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

    table_col_ids = [
        'eye', 'title', 'var_codes_filtered', 'long_name', 'platform_id', 'RI', 'time_period_start', 'time_period_end'
    ]
    table_col_names = [
        'Plot', 'Title', 'Variables', 'Station', 'Station code', 'RI', 'Start', 'End',
    ]
    table_columns = [{'name': name, 'id': i} for name, i in zip(table_col_names, table_col_ids)]
    # on rendering HTML snipplets in DataTable cells:
    # https://github.com/plotly/dash-table/pull/916
    table_columns[0]['presentation'] = 'markdown'  # column 'eye'
    table_columns[2]['presentation'] = 'markdown'  # column 'var_codes_filtered'
    table_columns[5]['presentation'] = 'markdown'  # column 'RI'

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
        # hidden_columns=['url', 'ecv_variables', 'ecv_variables_filtered', 'var_codes', 'platform_id_RI'],
        style_data={
            'whiteSpace': 'normal',
            'height': 'auto',
            'lineHeight': '15px'
        },
        style_cell={'textAlign': 'left'},
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
            dbc.CardFooter(reset_gantt_selection_button)
        ],
        style={'height': '70vh'}  # 70% of viewport height
    )

    datasets_table_card = dbc.Card([
        dbc.CardHeader(
            [
                html.B('b) Select your datasets here'), ' ', '(up to 10 datasets)'
            ]
            #style={'font-weight': 'bold'},
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
                                        multi=True,
                                        clearable=True,
                                    ),
                                ],
                                width=10
                            ),
                            dbc.Col(
                                children=dbc.Row(
                                    [
                                        dbc.Col(get_help_icon('#select-datasets'), width='auto'),
                                        dbc.Col(html.Div(get_next_button(SELECT_DATASETS_BUTTON_ID)), width='auto'),
                                    ],
                                    justify='end',
                                    align='center'
                                ),
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
