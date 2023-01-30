import dash_bootstrap_components as dbc
from dash import dcc

from utils import dash_dynamic_components as ddc, dash_persistence
from app_tabs.common import layout as common_layout


EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'exploratory-analysis-variables-checklist-all-none-switch'
EXPLORATORY_ANALYSIS_VARIABLES_CARDBODY_ROW_2_ID = 'exploratory-analysis-variables-cardbody-row-2'

EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ID = 'exploratory-analysis-variables-checklist'
EXPLORATORY_ANALYSIS_METHOD_RADIO_ID = 'exploratory-analysis-method-radio'
EXPLORATORY_ANALYSIS_METHOD_PARAMETERS_CARDBODY_ID = 'exploratory-analysis-method-parameters-cardbody'
EXPLORATORY_GRAPH_ID = 'exploratory-analysis-exploratory-graph'
EXPLORATORY_GRAPH_SCATTER_MODE_RADIO_ID = 'exploratory-analysis-graph-scatter-mode-radio'

GAUSSIAN_MEAN_AND_STD_METHOD = 'Gaussian mean and std'
PERCENTILES_METHOD = 'Percentiles'
MOVING_AVERAGE_METHOD = 'Moving average'
EXPLORATORY_ANALYSIS_METHODS = [
    GAUSSIAN_MEAN_AND_STD_METHOD,
    PERCENTILES_METHOD,
    MOVING_AVERAGE_METHOD,
]
EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_1_ID = 'exploratory-analysis-parameters-form-row-1'
EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_2_ID = 'exploratory-analysis-parameters-form-row-2'
EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_3_ID = 'exploratory-analysis-parameters-form-row-3'

AGGREGATION_PERIOD_RADIO_ID = 'exploratory-analysis-parameter-radio'
MIN_SAMPLE_SIZE_INPUT_ID = 'exploratory-analysis-min-sample-size'
SHOW_STD_SWITCH_ID = 'exploratory-analysis-show-std-switch'
STD_MODE_RADIO_ID = 'exploratory-analysis-std-mode-radio'
AGGREGATION_PERIOD_WORDINGS = {
    'D': ('day', 'daily'),
    'W': ('week', 'weekly'),
    'M': ('month', 'monthly'),
    'Q': ('season', 'seasonal'),
}
DEFAULT_AGGREGATION_PERIOD = 'M'

PERCENTILES_CHECKLIST_ID = 'data-analsysis-percentiles-checklist'
PERCENTILE_USER_DEF_INPUT_ID = 'data-analsysis-percentile-user-def-input'

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


def _get_exploratory_analysis_cardbody():
    select_all_none_variable_switch = dbc.Switch(
        id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID),
        label='Select all / none',
        # style={'margin-top': '10px'},
        value=True,
    )

    variables_checklist = dbc.Card([
        dbc.CardHeader('Variables'),
        dbc.CardBody([
            dbc.Row(select_all_none_variable_switch),
            dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_VARIABLES_CARDBODY_ROW_2_ID)),
        ]),
    ])

    analysis_method_radio = dbc.RadioItems(
        id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_METHOD_RADIO_ID),
        options=[
            {'label': analysis_method, 'value': analysis_method}
            for analysis_method in EXPLORATORY_ANALYSIS_METHODS
        ],
        value=EXPLORATORY_ANALYSIS_METHODS[0],
        inline=False,
        persistence=True,
        persistence_type='session',
    ),

    exploratory_analysis_method_radio = dbc.Card([
        dbc.CardHeader('Analysis method'),
        dbc.CardBody(analysis_method_radio),
    ])

    exploratory_analysis_method_parameters_card = dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody(
            dbc.Form([
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_1_ID)),
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_2_ID)),
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_3_ID)),
            ])
        ),
    ])

    return [
        dbc.Row(variables_checklist),
        dbc.Row(exploratory_analysis_method_radio),
        dbc.Row(exploratory_analysis_method_parameters_card),
    ]


exploratory_analysis_cardbody = _get_exploratory_analysis_cardbody()


def _get_exploratory_plot():
    graph = dcc.Graph(
        id=ddc.add_active_to_component_id(EXPLORATORY_GRAPH_ID),
        config=common_layout.GRAPH_CONFIG,
    ) # does it provide any performance improvement to scattergl?, config={'plotGlPixelRatio': 1})
    scatter_mode_radio = dbc.RadioItems(
        id=ddc.add_active_to_component_id(EXPLORATORY_GRAPH_SCATTER_MODE_RADIO_ID),
        options=[
            {'label': 'lines', 'value': 'lines'},
            {'label': 'markers', 'value': 'markers'},
            {'label': 'lines+markers', 'value': 'lines+markers'},
        ],
        value='lines',
        inline=True,
        persistence=True,
        persistence_type='session',
    )
    scatter_mode_input_group = [
        dbc.Label('Plot mode:', width='auto'),
        dbc.Col(scatter_mode_radio),
    ]

    return [
        dbc.Row(graph),
        dbc.Row(scatter_mode_input_group, align='end'),
    ]


exploratory_plot = _get_exploratory_plot()


aggregation_period_input = [
    dbc.Label('Aggregation period:'),
    dbc.RadioItems(
        id=ddc.add_active_to_component_id(AGGREGATION_PERIOD_RADIO_ID),
        options=[
            {'label': period_label, 'value': period_id}
            for period_id, (period_label, _) in AGGREGATION_PERIOD_WORDINGS.items()
        ],
        value=DEFAULT_AGGREGATION_PERIOD,
        **dash_persistence.get_dash_persistence_kwargs(True)
    )
]


minimal_sample_size_input = dbc.InputGroup(
    [
        dbc.InputGroupText('Minimal sample size for period:'),
        dbc.Input(
            id=ddc.add_active_to_component_id(MIN_SAMPLE_SIZE_INPUT_ID),
            type='number',
            min=1, step=1, value=5,
            debounce=True,
            **dash_persistence.get_dash_persistence_kwargs(True)
        ),
    ]
)


std_style_inputs = dbc.Row([
    dbc.Col(
        dbc.Switch(
            id=ddc.add_active_to_component_id(SHOW_STD_SWITCH_ID),
            label='Show standard deviation with',
            # style={'margin-top': '10px'},
            value=True,
            **dash_persistence.get_dash_persistence_kwargs(True)
        ),
        width='auto',
    ),
    dbc.Col(
        dbc.RadioItems(
            id=ddc.add_active_to_component_id(STD_MODE_RADIO_ID),
            options=[
                {'label': 'fill', 'value': 'fill'},
                {'label': 'error bars', 'value': 'error_bars'},
            ],
            value='fill',
            inline=True,
            **dash_persistence.get_dash_persistence_kwargs(True)
        ),
        width='auto',
    ),
])


def _get_percentiles_input_params():
    percentile_user_input = dbc.Input(
        id=ddc.add_active_to_component_id(PERCENTILE_USER_DEF_INPUT_ID),
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
                id=ddc.add_active_to_component_id(PERCENTILES_CHECKLIST_ID),
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
                **dash_persistence.get_dash_persistence_kwargs(True)
            ),
            dbc.InputGroup([
                dbc.InputGroupText('other'),
                percentile_user_input,
            ]),
        ]),
    ]


percentiles_input_params = _get_percentiles_input_params()
