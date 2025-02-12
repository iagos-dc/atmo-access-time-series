"""
ATMO-ACCESS time series service
"""

import os
os.environ['REACT_VERSION'] = '18.2.0'  # needed by dash_mantine_components

# Dash imports; for documentation (including tutorial), see: https://dash.plotly.com/
import dash
from dash import dcc, Dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import dash_mantine_components as dmc

import app_logging  # noq

# Local imports
from app_tabs.common.layout import get_app_data_stores, APP_TABS_ID, DATA_ANALYSIS_TAB_VALUE, \
    FILTER_DATA_TAB_VALUE, INFORMATION_TAB_VALUE
from app_tabs.information_tab.layout import get_information_tab
from app_tabs.search_datasets_tab.layout import get_search_datasets_tab
from app_tabs.select_datasets_tab.layout import SELECT_DATASETS_BUTTON_ID, \
    get_select_datasets_tab
from app_tabs.filter_data_tab.layout import FILTER_DATA_BUTTON_ID, \
    get_filter_data_tab
from app_tabs.data_analysis_tab.tabs_layout import get_data_analysis_tab
from utils.dash_persistence import get_dash_persistence_kwargs
from utils.exception_handler import alert_popups


# logos
ATMO_ACCESS_LOGO_FILENAME = 'atmo_access_logo.png'
ACTRIS_LOGO_FILENAME = 'actris_logo.png'
IAGOS_LOGO_FILENAME = 'iagos_logo.png'
ICOS_LOGO_FILENAME = 'icos_logo.png'


# Begin of definition of routines which constructs components of the dashboard

def get_dashboard_layout(app):
    stores = get_app_data_stores()

    feedback_button = dbc.Button(
        href='https://www.atmo-access.eu/virtual-access-feedback-form/#/',
        target='_blank',
        color='primary',
        outline=True,
        children=html.Div(
            [
                html.Div(
                    'Give feedback',
                    style={
                        'font-weight': 'bold',
                        'font-size': '135%',
                        'font-variant-caps': 'all-small-caps',
                        'white-space': 'nowrap'
                    }
                ),
            ],
        ),
        size='lg',
        # style={'height': '40px', 'align-self': 'center', 'margin-right': '20px'},
        style={'margin-right': '20px'},
    )

    # logo and application title
    title_and_logo_bar = html.Div(
        style={'display': 'flex', 'justify-content': 'space-between', 'align-content': 'center', 'margin-bottom': '10px'},
        children=[
            html.Div(children=[
                html.H2('Time-series analysis', style={'font-weight': 'bold'}),
            ]),
            html.Div(children=dbc.Row(
                [
                    dbc.Col(feedback_button),
                    dbc.Col(
                        html.A(
                            html.Img(
                                src=app.get_asset_url(ATMO_ACCESS_LOGO_FILENAME),
                                style={
                                    'float': 'right',
                                    'height': '70px',
                                    'margin-top': '10px'
                                }
                            ),
                            href="https://www.atmo-access.eu/",
                            target='_blank',
                        ),
                    )
                ],
            )),
        ]
    )

    app_tabs = dbc.Tabs(
        id=APP_TABS_ID,
        active_tab=INFORMATION_TAB_VALUE,
        children=[
            get_information_tab(
                actris_logo=app.get_asset_url(ACTRIS_LOGO_FILENAME),
                iagos_logo=app.get_asset_url(IAGOS_LOGO_FILENAME),
                icos_logo=app.get_asset_url(ICOS_LOGO_FILENAME),
            ),
            get_search_datasets_tab(),
            get_select_datasets_tab(),
            get_filter_data_tab(),
            get_data_analysis_tab(),
        ],
        style={'font-weight': 'bold'},
        #style={'font-size': '200%'},
        **get_dash_persistence_kwargs(persistence_id=True)
    )

    layout = html.Div(
        id='app-container-div',
        style={
            'margin': '10px',
            'padding-bottom': '50px'
        },
        children=stores + [
            html.Div(
                id='heading-div',
                className='twelve columns',
                children=[
                    title_and_logo_bar,
                    app_tabs,
                ]
            ),
            # error_message_popup,
        ] + alert_popups,
    )

    return layout

# End of definition of routines which constructs components of the dashboard


# Assign a dashboard layout to app Dash object
app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css'
    ],
    #prevent_initial_callbacks='initial_duplicate',
)

server = app.server

app.layout = dmc.MantineProvider(dbc.Container(
    get_dashboard_layout(app),
    fluid=True
))
app.title = 'ATMO-ACCESS time-series analysis'

# Begin of callback definitions and their helper routines.
# See: https://dash.plotly.com/basic-callbacks
# for a basic tutorial and
# https://dash.plotly.com/  -->  Dash Callback in left menu
# for more detailed documentation


# Launch the Dash application in development mode
if __name__ == "__main__":
    # in the new version of Dash (e.g. 2.18.1), it seems one to have to do this
    # (otherwise the env. variables HOST and PORT take precedence over the kwargs of app.run)
    #os.environ['HOST'] = 'localhost'
    #os.environ['PORT'] = '8050'
    #app.run(debug=True)
    app.run(port='8050', host='0.0.0.0', debug=True)
