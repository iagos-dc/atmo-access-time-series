from dash import dcc, html
import dash_bootstrap_components as dbc


DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'

VARIABLES_CHECKLIST_ID = 'data-analysis-variables-checklist'
# options
# value

RADIO_ID = 'data-analysis-parameter-radio'

GRAPH_ID = 'data-analysis-graph'


def get_variables_checklist():
    return dbc.Card([
        dbc.CardHeader('Variables'),
        dbc.CardBody(
            dbc.Checklist(
                id=VARIABLES_CHECKLIST_ID,
            ),
        ),
    ])


def get_analysis_method_radio():
    return dbc.Card([
        dbc.CardHeader('Analysis method'),
        dbc.CardBody(
            'here comes the analysis method choice...'
        ),
    ])


def get_analysis_parameters_card():
    return dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody(
            [
                dbc.Label('Aggregation period:'),
                dbc.RadioItems(
                    id='data-analysis-parameter-radio',
                    options=[
                        {'label': 'day', 'value': 'D'},
                        {'label': 'week', 'value': 'W'},
                        {'label': 'month', 'value': 'M'},
                        {'label': 'season', 'value': 'Q'},
                    ],
                    value='M',
                ),
            ]
        ),
    ])


def get_data_analysis_plot():
    return dcc.Graph(
        id=GRAPH_ID,
    )


def get_data_analysis_tab():
    return dcc.Tab(
        label='Data analysis',
        value=DATA_ANALYSIS_TAB_VALUE,
        children=html.Div(
            style={'margin': '20px'},
            children=dbc.Container(
                dbc.Row([
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
                ]),
                fluid=True,
            )
        )
    )
