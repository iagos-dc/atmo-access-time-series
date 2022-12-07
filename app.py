"""
ATMO-ACCESS time series service
"""

# import gunicorn

# Dash imports; for documentation (including tutorial), see: https://dash.plotly.com/
import dash
from dash import dcc, Dash
from dash import html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc


# Local imports
from app_tabs.common.layout import get_app_data_stores
from app_tabs.search_datasets_tab.layout import SEARCH_DATASETS_TAB_VALUE, SEARCH_DATASETS_BUTTON_ID, \
    get_search_datasets_tab
from app_tabs.select_datasets_tab.layout import SELECT_DATASETS_TAB_VALUE, SELECT_DATASETS_BUTTON_ID, \
    get_select_datasets_tab
from app_tabs.filter_data_tab.layout import FILTER_DATA_TAB_VALUE, FILTER_DATA_BUTTON_ID, \
    get_filter_data_tab
from app_tabs.data_analysis_tab.layout import DATA_ANALYSIS_TAB_VALUE, \
    get_data_analysis_tab


# Configuration of the app
# See: https://dash.plotly.com/devtools#configuring-with-run_server
# for the usual Dash app, and:
# https://github.com/plotly/jupyter-dash/blob/master/notebooks/getting_started.ipynb
# for a JupyterDash app version.
# app_conf = {'mode': 'external', 'debug': True}  # for running inside a Jupyter notebook change 'mode' to 'inline'
# RUNNING_IN_BINDER = os.environ.get('BINDER_SERVICE_HOST') is not None
# if RUNNING_IN_BINDER:
#     JupyterDash.infer_jupyter_proxy_config()
# else:
#     app_conf.update({'host': 'localhost', 'port': 9235})


# Below there are id's of Dash JS components.
# The components themselves are declared in the dashboard layout (see the function get_dashboard_layout).
# Essential properties of each component are explained in the comments below.
APP_TABS_ID = 'app-tabs'    # see: https://dash.plotly.com/dash-core-components/tabs; method 1 (content as callback)
    # value contains an id of the active tab
    # children contains a list of layouts of each tab

# Atmo-Access logo url
ATMO_ACCESS_LOGO_URL = \
    'https://www7.obs-mip.fr/wp-content-aeris/uploads/sites/82/2021/03/ATMO-ACCESS-Logo-final_horizontal-payoff-grey-blue.png'


# Begin of definition of routines which constructs components of the dashboard

def get_dashboard_layout():
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
                        #src=app.get_asset_url('atmo_access_logo.png') if not RUNNING_IN_BINDER else ATMO_ACCESS_LOGO_URL,
                        src=ATMO_ACCESS_LOGO_URL,
                        style={'float': 'right', 'height': '70px', 'margin-top': '10px'}
                    ),
                    href="https://www.atmo-access.eu/",
                ),
            ]),
        ]
    )

    app_tabs = dcc.Tabs(
        id=APP_TABS_ID,
        value=SEARCH_DATASETS_TAB_VALUE,
        children=[
            get_search_datasets_tab(),
            get_select_datasets_tab(),
            get_filter_data_tab(),
            get_data_analysis_tab(),
        ]
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
            )
        ]
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
)

server = app.server


app.layout = get_dashboard_layout()

# Begin of callback definitions and their helper routines.
# See: https://dash.plotly.com/basic-callbacks
# for a basic tutorial and
# https://dash.plotly.com/  -->  Dash Callback in left menu
# for more detailed documentation


@app.callback(
    Output(APP_TABS_ID, 'value'),
    Input(SEARCH_DATASETS_BUTTON_ID, 'n_clicks'),
    Input(SELECT_DATASETS_BUTTON_ID, 'n_clicks'),
    Input(FILTER_DATA_BUTTON_ID, 'n_clicks')
)
def change_app_tab(search_datasets_button_clicks, select_datasets_button_clicks, filter_data_button_clicks):
    # trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    trigger = dash.ctx.triggered_id
    if trigger == SEARCH_DATASETS_BUTTON_ID:
        return SELECT_DATASETS_TAB_VALUE
    elif trigger == SELECT_DATASETS_BUTTON_ID:
        return FILTER_DATA_TAB_VALUE
    elif trigger == FILTER_DATA_BUTTON_ID:
        return DATA_ANALYSIS_TAB_VALUE
    else:
        return SEARCH_DATASETS_TAB_VALUE


# Launch the Dash application.
# app_conf['debug'] = False
if __name__ == "__main__":
    app.run_server(debug=True, host='0.0.0.0', port=8050)
