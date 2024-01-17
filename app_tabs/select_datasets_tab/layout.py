import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

from app_tabs.common.layout import SELECT_DATASETS_TAB_VALUE, NON_INTERACTIVE_GRAPH_CONFIG, get_next_button


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

    reset_gantt_selection_button = dbc.Button(
        id=RESET_DATASETS_SELECTION_BUTTON_ID,
        n_clicks=0,
        color='danger',
        type='submit',
        style={'font-weight': 'bold'},
        children='Clear',
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

    gantt_diagram_card = dbc.Card([
        dbc.CardHeader(
            [
                html.B('Datasets time coverage'),
                html.P('Select groups of datasets to fill in the table on the right'),
            ],
            # style={'display': 'flex'}
        ),
        dbc.CardBody([
            dbc.Row([
                dbc.Col(reset_gantt_selection_button, width='auto'),
                dbc.Col(gantt_view_radio, width='auto'),
            ], justify='between', style={'margin-bottom': '10px'}),
            dbc.Row(dbc.Col(html.Div(gantt_graph))),
        ])
    ])

    datasets_table_card = dbc.Card([
        dbc.CardHeader(
            'Select your datasets here',
            style={'font-weight': 'bold'},
        ),
        dbc.CardBody([
            dbc.Row(dbc.Col(all_none_switch)),
            dbc.Row(dbc.Col(table)),
        ])
    ])

    return dbc.Tab(
        label='3. Select datasets',
        id=SELECT_DATASETS_TAB_VALUE,
        tab_id=SELECT_DATASETS_TAB_VALUE,
        disabled=True,

        children=[
            html.Div(
                style={'margin-top': '5px', 'margin-left': '20px', 'margin-right': '20px'},
                children=[
                    dbc.Row(
                        dbc.Col(
                            children=html.Div(get_next_button(SELECT_DATASETS_BUTTON_ID)),
                            width='auto',
                        ),
                        justify='end',
                        style={'margin-bottom': '10px'},
                    ),
                    dbc.Row([
                        dbc.Col(
                            width=5,
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
