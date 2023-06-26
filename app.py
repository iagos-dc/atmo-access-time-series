"""
ATMO-ACCESS time series service
"""

# Dash imports; for documentation (including tutorial), see: https://dash.plotly.com/
import dash
from dash import dcc, Dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


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
from log import log_exception
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

    # logo and application title
    title_and_logo_bar = html.Div(
        style={'display': 'flex', 'justify-content': 'space-between', 'margin-bottom': '20px'},
        children=[
            html.Div(children=[
                html.H2('Time-series analysis', style={'font-weight': 'bold'}),
            ]),
            html.Div(children=[
                html.A(
                    html.Img(
                        src=app.get_asset_url(ATMO_ACCESS_LOGO_FILENAME),
                        style={'float': 'right', 'height': '70px', 'margin-top': '10px'}
                    ),
                    href="https://www.atmo-access.eu/",
                    target='_blank',
                ),
            ]),
        ]
    )

    app_tabs = dcc.Tabs(
        id=APP_TABS_ID,
        value=INFORMATION_TAB_VALUE,
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
        **get_dash_persistence_kwargs(persistence_id=True)
    )

    layout = html.Div(
        id='app-container-div',
        style={'margin': '30px', 'padding-bottom': '50px'},
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
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
    #prevent_initial_callbacks='initial_duplicate',
)

server = app.server

app.layout = get_dashboard_layout(app)
app.title = 'ATMO-ACCESS time-series analysis'

# Begin of callback definitions and their helper routines.
# See: https://dash.plotly.com/basic-callbacks
# for a basic tutorial and
# https://dash.plotly.com/  -->  Dash Callback in left menu
# for more detailed documentation


# Launch the Dash application in development mode
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
