from dash import dcc, html
import dash_bootstrap_components as dbc
from utils import combo_input_AIO, dash_dynamic_components as ddc, dash_persistence


DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'
KIND_OF_ANALYSIS_TABS_ID = 'kind-of-analysis-tabs'
DATA_ANALYSIS_PARAMETERS_CARDBODY_ID = 'data-analysis-parameters-cardbody'
EXPLORATORY_ANALYSIS_TAB_ID = 'exploratory-analysis'
TREND_ANALYSIS_TAB_ID = 'trend-analysis'
MULTIVARIATE_ANALYSIS_TAB_ID = 'multivariate-analysis'

VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'data-analysis-variables-checklist-all-none-switch'
VARIABLES_CHECKLIST_ID = 'data-analysis-variables-checklist'

MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ID = 'multivariate-analysis-variables-cardbody'
X_VARIABLE_SELECT_ID = 'x-variable-select'
Y_VARIABLE_SELECT_ID = 'y-variable-select'
C_VARIABLE_SELECT_ID = 'c-variable-select'

EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID = 'exploratory-analysis-inputs-group'
DATA_ANALYSIS_METHOD_RADIO_ID = 'data-analysis-method-radio'
DATA_ANALYSIS_METHOD_PARAMETERS_CARDBODY_ID = 'data-analysis-method-parameters-cardbody'

GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID = 'data-analysis-gaussian-mean-and-std-combo-input-aio'
PERCENTILES_COMBO_INPUT_AIO_ID = 'data-analysis-percentiles-combo-input-aio'

AGGREGATION_PERIOD_RADIO_ID = 'data-analysis-parameter-radio'
MIN_SAMPLE_SIZE_INPUT_ID = 'data-analysis-min-sample-size'
PERCENTILES_CHECKLIST_ID = 'data-analsysis-percentiles-checklist'
PERCENTILE_USER_DEF_INPUT_ID = 'data-analsysis-percentile-user-def-input'
SHOW_STD_SWITCH_ID = 'data-analysis-show-std-switch'
STD_MODE_RADIO_ID = 'data-analysis-std-mode-radio'

SCATTER_PLOT_PARAMS_RADIO_ID = 'scatter-plot-param-radio'
SCATTER_PLOT_PARAM_INDIVIDUAL_OBSERVATIONS = 'scatter-plot-param-individual-observations'
SCATTER_PLOT_PARAM_HEXBIN = 'scatter-plot-param-hexbin'

DATA_ANALYSIS_FIGURE_CONTAINER_ID = 'data-analysis-figure-container'

EXPLORATORY_GRAPH_ID = 'data-analysis-exploratory-graph'
GRAPH_SCATTER_MODE_RADIO_ID = 'data-analysis-graph-scatter-mode-radio'

MULTIVARIATE_GRAPH_ID = 'data-analysis-multivariate-graph'

GAUSSIAN_MEAN_AND_STD_METHOD = 'Gaussian mean and std'
PERCENTILES_METHOD = 'Percentiles'
MOVING_AVERAGES_METHOD = 'Moving average'
AUTOCORRELATION_METHOD = 'Autocorrelation'

SCATTER_PLOT_METHOD = 'Scatter plot'
LINEAR_REGRESSION_METHOD = 'Linear regression'

ANALYSIS_METHOD_LABELS_BY_KIND_OF_ANALYSIS_TABS_ID = {
    EXPLORATORY_ANALYSIS_TAB_ID: [
        GAUSSIAN_MEAN_AND_STD_METHOD,
        PERCENTILES_METHOD,
        MOVING_AVERAGES_METHOD,
        AUTOCORRELATION_METHOD,
    ],
    TREND_ANALYSIS_TAB_ID: [
    ],
    MULTIVARIATE_ANALYSIS_TAB_ID: [
        SCATTER_PLOT_METHOD,
        LINEAR_REGRESSION_METHOD,
    ],
}

AGGREGATION_PERIOD_WORDINGS = {
    'D': ('day', 'daily'),
    'W': ('week', 'weekly'),
    'M': ('month', 'monthly'),
    'Q': ('season', 'seasonal'),
}
DEFAULT_AGGREGATION_PERIOD = 'M'


# globals
gaussian_mean_and_std_parameters_combo_input = None
percentiles_parameters_combo_input = None


def get_variables_checklist():
    select_all_none_variable_switch = dbc.Switch(
        id=ddc.add_active_to_component_id(VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID),
        label='Select all / none',
        # style={'margin-top': '10px'},
        value=True,
    )

    return dbc.Card([
        dbc.CardHeader('Variables'),
        dbc.CardBody([
            select_all_none_variable_switch,
            dbc.Checklist(
                id=ddc.add_active_to_component_id(VARIABLES_CHECKLIST_ID),
            ),
        ]),
    ])


def get_message_not_enough_variables_for_multivariate_analysis():
    return 'For multivariate analysis choose at least 2 variables'


def get_variable_dropdown(dropdown_id, axis_label, options, value, persistence_id=None):
    persistence_kwargs = dash_persistence.get_dash_persistence_kwargs(persistence_id)

    return dbc.InputGroup([
        dbc.InputGroupText(axis_label),
        dbc.Select(
            id=dropdown_id,
            options=options,
            value=value,
            **persistence_kwargs,
        ),
    ])


def get_analysis_method_radio():
    analysis_method_radio = dbc.RadioItems(
        inline=False,
        id=ddc.add_active_to_component_id(DATA_ANALYSIS_METHOD_RADIO_ID),
#        persistence=True,
#        persistence_type='session',
    ),

    return dbc.Card([
        dbc.CardHeader('Analysis method'),
        dbc.CardBody(analysis_method_radio),
    ])


def get_analysis_method_parameters_card():
    return dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody(
            id=ddc.add_active_to_component_id(DATA_ANALYSIS_METHOD_PARAMETERS_CARDBODY_ID),
        ),
    ])


def _get_aggregation_period_input(id_):
    return [
        dbc.Label('Aggregation period:'),
        dbc.RadioItems(
            id=id_,
            options=[
                {'label': period_label, 'value': period_id}
                for period_id, (period_label, _) in AGGREGATION_PERIOD_WORDINGS.items()
            ],
            value=DEFAULT_AGGREGATION_PERIOD,
#            persistence=True,
#            persistence_type='session',
        )
    ]


def _get_minimal_sample_size_input(id_):
    return [
        dbc.InputGroup(
            [
                dbc.InputGroupText('Minimal sample size for period:'),
                dbc.Input(
                    id=id_,
                    type='number',
                    min=1, step=1, value=5,
                    debounce=True,
#                    persistence=True,
#                    persistence_type='session',
                ),
            ]
        )
    ]


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


def get_gaussian_mean_and_std_parameters_combo_input(parent_component):
    std_style_inputs = dbc.Row([
        dbc.Col(
            dbc.Switch(
                id=SHOW_STD_SWITCH_ID,
                label='Show standard deviation with',
                # style={'margin-top': '10px'},
                value=True,
#                persistence=True,
#                persistence_type='session',
            ),
            width='auto',
        ),
        dbc.Col(
            dbc.RadioItems(
                id=STD_MODE_RADIO_ID,
                options=[
                    {'label': 'fill', 'value': 'fill'},
                    {'label': 'error bars', 'value': 'error_bars'},
                ],
                value='fill',
                inline=True,
#                persistence=True,
#                persistence_type='session',
            ),
            width='auto',
        ),
    ])

    combo_inputs = [
        *_get_aggregation_period_input(AGGREGATION_PERIOD_RADIO_ID),
        *_get_minimal_sample_size_input(MIN_SAMPLE_SIZE_INPUT_ID),
        std_style_inputs,
    ]

    return combo_input_AIO.ComboInputAIO(
        children=combo_inputs,
        parent_component=parent_component,
        group_id=EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID,
        aio_id=GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID,
        input_component_ids=[AGGREGATION_PERIOD_RADIO_ID, MIN_SAMPLE_SIZE_INPUT_ID, SHOW_STD_SWITCH_ID, STD_MODE_RADIO_ID]
    )


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


def get_scatter_plot_parameters():
    return [
        dbc.Label('Plot type:'),
        dbc.RadioItems(
            id=ddc.add_active_to_component_id(SCATTER_PLOT_PARAMS_RADIO_ID),
            options=[
                {'label': 'hexagonal bins', 'value': SCATTER_PLOT_PARAM_HEXBIN},
                {'label': 'individual observations', 'value': SCATTER_PLOT_PARAM_INDIVIDUAL_OBSERVATIONS},
            ],
            value=SCATTER_PLOT_PARAM_INDIVIDUAL_OBSERVATIONS,
#            persistence=True,
#            persistence_type='session',
        )
    ]


def get_exploratory_plot():
    graph = dcc.Graph(id=ddc.add_active_to_component_id(EXPLORATORY_GRAPH_ID)) # does it provide any performance improvement to scattergl?, config={'plotGlPixelRatio': 1})
    scatter_mode_radio = dbc.RadioItems(
        id=ddc.add_active_to_component_id(GRAPH_SCATTER_MODE_RADIO_ID),
        options=[
            {'label': 'lines', 'value': 'lines'},
            {'label': 'markers', 'value': 'markers'},
            {'label': 'lines+markers', 'value': 'lines+markers'},
        ],
        value='lines',
        inline=True,
#        persistence=True,
#        persistence_type='session',
    )
    scatter_mode_input_group = [
        dbc.Label('Plot mode:', width='auto'),
        dbc.Col(scatter_mode_radio),
    ]

    return [
        dbc.Row(graph),
        dbc.Row(scatter_mode_input_group, align='end'),
    ]


def get_multivariate_plot():
    graph = dcc.Graph(id=ddc.add_active_to_component_id(MULTIVARIATE_GRAPH_ID)) # does it provide any performance improvement to scattergl?, config={'plotGlPixelRatio': 1})
    return dbc.Row(graph)


def get_exploratory_analysis_parameters():
    return [
        dbc.Row(get_variables_checklist()),
        dbc.Row(get_analysis_method_radio()),
        dbc.Row(get_analysis_method_parameters_card()),
    ]


def get_multivariate_analysis_parameters():
    return [
        dbc.Row(
            dbc.Card([
                dbc.CardHeader('Variables'),
                dbc.CardBody(id=ddc.add_active_to_component_id(MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ID)),
            ]),
        ),
        dbc.Row(get_analysis_method_radio()),
        dbc.Row(get_analysis_method_parameters_card()),
    ]


def get_data_analysis_tab():
    global gaussian_mean_and_std_parameters_combo_input, gaussian_mean_and_std_extra_graph_controllers_combo_input
    global percentiles_parameters_combo_input

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
#                            persistence=True,
#                            persistence_type='session',
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
        ]
    )

    gaussian_mean_and_std_parameters_combo_input = get_gaussian_mean_and_std_parameters_combo_input(data_analysis_tab)
    percentiles_parameters_combo_input = get_percentiles_parameters_combo_input(data_analysis_tab)

    return data_analysis_tab
