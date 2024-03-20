from dash import dcc, html
import dash_bootstrap_components as dbc

from app_tabs.common.layout import DATA_ANALYSIS_TAB_VALUE, get_help_icon


KIND_OF_ANALYSIS_TABS_ID = 'kind-of-analysis-tabs'
DATA_ANALYSIS_PARAMETERS_CARDBODY_ID = 'data-analysis-parameters-cardbody'
DATA_ANALYSIS_FIGURE_CONTAINER_ID = 'data-analysis-figure-container'

EXPLORATORY_ANALYSIS_TAB_ID = 'exploratory-analysis'
TREND_ANALYSIS_TAB_ID = 'trend-analysis'
MULTIVARIATE_ANALYSIS_TAB_ID = 'multivariate-analysis'

BORDER_STYLE = {'border-style': 'solid', 'border-width': '1px', 'border-color': 'lightgrey'}


def get_data_analysis_tab():
    data_analysis_tab_container_content = dbc.Row([
        dbc.Col(
            children=dbc.Container(
                [
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
                    # TODO: implement download data button
                    # html.Div(dbc.Row(
                    #     [
                    #         dbc.Col(dbc.Button('Download data'), width=4, align='left'),
                    #         dbc.Col(dbc.Button('Download figures'), width=4, align='right'),
                    #     ],
                    #     align='bottom',
                    #     justify='between',
                    # ))
                ],
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

    data_analysis_tab = dbc.Tab(
        label='4. Data analysis',
        id=DATA_ANALYSIS_TAB_VALUE,
        tab_id=DATA_ANALYSIS_TAB_VALUE,
        #value=DATA_ANALYSIS_TAB_VALUE,
        disabled=True,
        children=[
            html.Div(
                style={
                    'margin-top': '10px',
                    'margin-left': '0px',
                    'margin-right': '0px',
                },
                children=[
                    dbc.Row(
                        dbc.Col(get_help_icon('#data-analysis'), width='auto'),
                        justify='end',
                        style={'margin-bottom': '10px'},
                    ),
                    data_analysis_tab_container_content
                ],
        # children=dbc.Container(
        #     style={
        #         'margin-top': '10px',
        #         'margin-left': '0px',
        #         'margin-right': '0px',
        #     },
        #     children=data_analysis_tab_container_content,
        #     fluid=True,
        # )
            ),
        ],
    )

    return data_analysis_tab
