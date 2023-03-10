import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

SELECT_DATASETS_TAB_VALUE = 'select-datasets-tab'

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


def get_select_datasets_tab():
    return dcc.Tab(label='Select datasets', value=SELECT_DATASETS_TAB_VALUE,
                                  children=html.Div(style={'margin': '20px'}, children=[
        html.Div(id='select-datasets-left-panel-div', className='four columns', children=[
            html.Div(id='select-datasets-left-left-subpanel-div', className='nine columns', children=
                dbc.RadioItems(
                    id=GANTT_VIEW_RADIO_ID,
                    options=[
                        {'label': 'compact view', 'value': 'compact'},
                        {'label': 'detailed view', 'value': 'detailed'},
                    ],
                    value='compact',
                    inline=True)),
            html.Div(id='select-datasets-left-right-subpanel-div', className='three columns', children=
                dbc.Button(id=SELECT_DATASETS_BUTTON_ID, n_clicks=0,
                       color='primary', type='submit',
                       style={'font-weight': 'bold'},
                       children='Select datasets'))
        ]),
        html.Div(id='select-datasets-right-panel-div', className='eight columns', children=None),
        html.Div(id='select-datasets-main-panel-div', className='twelve columns', children=[
            dcc.Graph(
                id=GANTT_GRAPH_ID,
            ),
            dbc.Switch(
                id=DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID,
                label='Select all / none',
                style={'margin-top': '10px'},
                value=False,
            ),
            dash_table.DataTable(
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
            ),
            html.Div(id=QUICKLOOK_POPUP_ID),
        ]),
    ]))
