from dash import dcc, html
import dash_bootstrap_components as dbc


DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'

VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'data-analysis-variables-checklist-all-none-switch'
VARIABLES_CHECKLIST_ID = 'data-analysis-variables-checklist'
# options
# value

ANALYSIS_METHOD_RADIO_ID = 'data-analysis-method-radio'

DATA_ANALYSIS_SPECIFICATION_STORE_ID = 'data-analysis-specification-store'

RADIO_ID = 'data-analysis-parameter-radio'

MIN_SAMPLE_SIZE_INPUT_ID = 'data-analysis-min-sample-size'

SHOW_STD_SWITCH_ID = 'data-analysis-show-std-switch'

GRAPH_ID = 'data-analysis-graph'


GAUSSIAN_MEAN_AND_STD_METHOD = 'Gaussian mean and std'
PERCENTILES_METHOD = 'Percentiles'
MOVING_AVERAGES_METHOD = 'Moving average'
AUTOCORRELATION_METHOD = 'Autocorrelation'
ANALYSIS_METHOD_LABELS = [
    GAUSSIAN_MEAN_AND_STD_METHOD,
    PERCENTILES_METHOD,
    MOVING_AVERAGES_METHOD,
    AUTOCORRELATION_METHOD,
]


def get_variables_checklist():
    select_all_none_variable_switch = dbc.Switch(
        id=VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID,
        label='Select all / none',
        # style={'margin-top': '10px'},
        value=True,
    )

    return dbc.Card([
        dbc.CardHeader('Variables'),
        dbc.CardBody([
            select_all_none_variable_switch,
            dbc.Checklist(
                id=VARIABLES_CHECKLIST_ID,
            ),
        ]),
    ])


def get_analysis_method_radio():
    return dbc.Card([
        dbc.CardHeader('Analysis method'),
        dbc.CardBody(
            dbc.RadioItems(
                options=[{'label': analysis_method, 'value': analysis_method} for analysis_method in ANALYSIS_METHOD_LABELS],
                value='Gaussian mean and std',
                inline=False,
                id=ANALYSIS_METHOD_RADIO_ID,
            ),
        ),
    ])


def get_analysis_parameters_card():
    return dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody(
            [
                dbc.Label('Aggregation period:'),
                dbc.RadioItems(
                    id=RADIO_ID,
                    options=[
                        {'label': 'day', 'value': 'D'},
                        {'label': 'week', 'value': 'W'},
                        {'label': 'month', 'value': 'M'},
                        {'label': 'season', 'value': 'Q'},
                    ],
                    value='M',
                ),
                dbc.InputGroup(
                    [
                        dbc.InputGroupText('Minimal sample size for period:'),
                        dbc.Input(type='number', min=1, step=1, value=5, id=MIN_SAMPLE_SIZE_INPUT_ID),
                    ]
                ), 
                dbc.Switch(
                    id=SHOW_STD_SWITCH_ID,
                    label='Calculate standard deviation',
                    # style={'margin-top': '10px'},
                    value=False,
                )
            ]
        ),
    ])


def get_data_analysis_plot():
    return dcc.Graph(
        id=GRAPH_ID,
    )


def get_data_analysis_tab():
    data_analysis_tab_container_content = dbc.Row([
        dbc.Col(
            children=dbc.Container(
                dbc.Row([
                   get_variables_checklist(),
                   get_analysis_method_radio(),
                   get_analysis_parameters_card(),
                ]),
                fluid=True,
            ),
            width=4,
        ),
        dbc.Col(
            children=dbc.Container(
                dbc.Row([
                    get_data_analysis_plot(),
                ]),
                fluid=True,
            ),
            width=8),
    ])

    return dcc.Tab(
        label='Data analysis',
        value=DATA_ANALYSIS_TAB_VALUE,
        children=[
            dcc.Store(id=DATA_ANALYSIS_SPECIFICATION_STORE_ID),
            html.Div(
                style={'margin': '20px'},
                children=dbc.Container(
                    data_analysis_tab_container_content,
                    fluid=True,
                )
            ),
        ]
    )
