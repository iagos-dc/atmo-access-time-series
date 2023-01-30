from dash import dcc, html
import dash_bootstrap_components as dbc

DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'
KIND_OF_ANALYSIS_TABS_ID = 'kind-of-analysis-tabs'
DATA_ANALYSIS_PARAMETERS_CARDBODY_ID = 'data-analysis-parameters-cardbody'
DATA_ANALYSIS_FIGURE_CONTAINER_ID = 'data-analysis-figure-container'

EXPLORATORY_ANALYSIS_TAB_ID = 'exploratory-analysis'
TREND_ANALYSIS_TAB_ID = 'trend-analysis'
MULTIVARIATE_ANALYSIS_TAB_ID = 'multivariate-analysis'


def get_data_analysis_tab():
    data_analysis_tab_container_content = dbc.Row([
        dbc.Col(
            children=dbc.Container(
                dbc.Card([
                    dbc.CardHeader(
                        dbc.Tabs(
                            [
                                dbc.Tab(label='Exploratory analysis', tab_id=EXPLORATORY_ANALYSIS_TAB_ID),
                                dbc.Tab(label='Trend analysis', tab_id=TREND_ANALYSIS_TAB_ID),
                                dbc.Tab(label='Multivariate analysis', tab_id=MULTIVARIATE_ANALYSIS_TAB_ID),
                            ],
                            id=KIND_OF_ANALYSIS_TABS_ID,
                            active_tab=EXPLORATORY_ANALYSIS_TAB_ID,
                            persistence=True,
                            persistence_type='session',
                        )
                    ),
                    dbc.CardBody(
                        id=DATA_ANALYSIS_PARAMETERS_CARDBODY_ID,
                    ),
                ]),
                fluid=True,
            ),
            width=4,
        ),
        dbc.Col(
            children=dbc.Container(
                id=DATA_ANALYSIS_FIGURE_CONTAINER_ID,
                fluid=True,
            ),
            width=8),
    ])

    data_analysis_tab = dcc.Tab(
        label='Data analysis',
        value=DATA_ANALYSIS_TAB_VALUE,
        children=[
            html.Div(
                style={'margin': '20px'},
                children=dbc.Container(
                    data_analysis_tab_container_content,
                    fluid=True,
                )
            ),
        ],
    )

    return data_analysis_tab
