from dash import dcc, html
import dash_bootstrap_components as dbc

from app_tabs.data_analysis_tab.exploratory_analysis_layout import EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID, \
    AGGREGATION_PERIOD_RADIO_ID, MIN_SAMPLE_SIZE_INPUT_ID, _get_aggregation_period_input, _get_minimal_sample_size_input
from utils import combo_input_AIO, dash_persistence


DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'
KIND_OF_ANALYSIS_TABS_ID = 'kind-of-analysis-tabs'
DATA_ANALYSIS_PARAMETERS_CARDBODY_ID = 'data-analysis-parameters-cardbody'
DATA_ANALYSIS_FIGURE_CONTAINER_ID = 'data-analysis-figure-container'

EXPLORATORY_ANALYSIS_TAB_ID = 'exploratory-analysis'
TREND_ANALYSIS_TAB_ID = 'trend-analysis'
MULTIVARIATE_ANALYSIS_TAB_ID = 'multivariate-analysis'

PERCENTILES_COMBO_INPUT_AIO_ID = 'data-analysis-percentiles-combo-input-aio'

PERCENTILES_CHECKLIST_ID = 'data-analsysis-percentiles-checklist'
PERCENTILE_USER_DEF_INPUT_ID = 'data-analsysis-percentile-user-def-input'

GAUSSIAN_MEAN_AND_STD_METHOD = 'Gaussian mean and std'
PERCENTILES_METHOD = 'Percentiles'
MOVING_AVERAGES_METHOD = 'Moving average'
AUTOCORRELATION_METHOD = 'Autocorrelation'

ANALYSIS_METHOD_LABELS_BY_KIND_OF_ANALYSIS_TABS_ID = {
    EXPLORATORY_ANALYSIS_TAB_ID: [
        GAUSSIAN_MEAN_AND_STD_METHOD,
        PERCENTILES_METHOD,
        MOVING_AVERAGES_METHOD,
        AUTOCORRELATION_METHOD,
    ],
    TREND_ANALYSIS_TAB_ID: [],
    MULTIVARIATE_ANALYSIS_TAB_ID: [],
}

# globals
gaussian_mean_and_std_parameters_combo_input = None
percentiles_parameters_combo_input = None


def get_variable_dropdown(dropdown_id, axis_label, options, value, disabled=False, persistence_id=None):
    persistence_kwargs = dash_persistence.get_dash_persistence_kwargs(persistence_id)

    return dbc.InputGroup([
        dbc.InputGroupText(axis_label),
        dbc.Select(
            id=dropdown_id,
            options=options,
            value=value,
            disabled=disabled,
            **persistence_kwargs,
        ),
    ])


LINE_DASH_STYLE_BY_PERCENTILE = {
    'min': 'dot',
    '5': 'dash',
    '25': 'dashdot',
    '50': 'solid',
    '75': 'dashdot',
    '95': 'dash',
    'max': 'dot',
    'other': 'longdashdot'
}

def _get_percentiles_checklist(percentiles_checklist_id, percentile_user_input_id):
    percentile_user_input = dbc.Input(
        id=percentile_user_input_id,
        type='number',
        placeholder='percentile',
        min=0, max=100, step=None, value=None,
        debounce=True,
    )

    return [
        dbc.Row(
            dbc.Col(
                dbc.Label('Percentiles'),
            ),
        ),
        dbc.Row([
            dbc.Checklist(
                id=percentiles_checklist_id,
                options=[
                    {'label': 'min', 'value': 0},
                    {'label': 5, 'value': 5},
                    {'label': 25, 'value': 25},
                    {'label': 50, 'value': 50},
                    {'label': 75, 'value': 75},
                    {'label': 95, 'value': 95},
                    {'label': 'max', 'value': 100},
                ],
                value=[5, 50, 95],
                inline=True,
#                persistence=True,
#                persistence_type='session',
            ),
            dbc.InputGroup([
                dbc.InputGroupText('other'),
                percentile_user_input,
            ]),
        ]),
    ]


def get_percentiles_parameters_combo_input(parent_component):
    combo_inputs = [
        *_get_aggregation_period_input(AGGREGATION_PERIOD_RADIO_ID),
        *_get_minimal_sample_size_input(MIN_SAMPLE_SIZE_INPUT_ID),
        *_get_percentiles_checklist(
            PERCENTILES_CHECKLIST_ID,
            PERCENTILE_USER_DEF_INPUT_ID,
        ),
    ]

    return combo_input_AIO.ComboInputAIO(
        children=combo_inputs,
        parent_component=parent_component,
        group_id=EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID,
        aio_id=PERCENTILES_COMBO_INPUT_AIO_ID,
        input_component_ids=[
            AGGREGATION_PERIOD_RADIO_ID,
            MIN_SAMPLE_SIZE_INPUT_ID,
            PERCENTILES_CHECKLIST_ID,
            PERCENTILE_USER_DEF_INPUT_ID,
        ]
    )


def get_data_analysis_tab():
    #global gaussian_mean_and_std_parameters_combo_input, gaussian_mean_and_std_extra_graph_controllers_combo_input
    #global percentiles_parameters_combo_input

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
    #gaussian_mean_and_std_parameters_combo_input = get_gaussian_mean_and_std_parameters_combo_input(data_analysis_tab)
    #percentiles_parameters_combo_input = get_percentiles_parameters_combo_input(data_analysis_tab)

    return data_analysis_tab
