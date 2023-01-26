import dash_bootstrap_components as dbc
from dash import dcc

from utils import dash_dynamic_components as ddc, combo_input_AIO

EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID = 'data-analysis-variables-checklist-all-none-switch'
EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ID = 'data-analysis-variables-checklist'
EXPLORATORY_ANALYSIS_METHOD_RADIO_ID = 'data-analysis-method-radio'
EXPLORATORY_ANALYSIS_METHOD_PARAMETERS_CARDBODY_ID = 'data-analysis-method-parameters-cardbody'
EXPLORATORY_GRAPH_ID = 'data-analysis-exploratory-graph'
EXPLORATORY_GRAPH_SCATTER_MODE_RADIO_ID = 'data-analysis-graph-scatter-mode-radio'

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
EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_4_ID = 'exploratory-analysis-parameters-form-row-4'

EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID = 'exploratory-analysis-inputs-group'
GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID = 'data-analysis-gaussian-mean-and-std-combo-input-aio'
AGGREGATION_PERIOD_RADIO_ID = 'data-analysis-parameter-radio'
MIN_SAMPLE_SIZE_INPUT_ID = 'data-analysis-min-sample-size'
SHOW_STD_SWITCH_ID = 'data-analysis-show-std-switch'
STD_MODE_RADIO_ID = 'data-analysis-std-mode-radio'
AGGREGATION_PERIOD_WORDINGS = {
    'D': ('day', 'daily'),
    'W': ('week', 'weekly'),
    'M': ('month', 'monthly'),
    'Q': ('season', 'seasonal'),
}
DEFAULT_AGGREGATION_PERIOD = 'M'


def get_variables_checklist():
    select_all_none_variable_switch = dbc.Switch(
        id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID),
        label='Select all / none',
        # style={'margin-top': '10px'},
        value=True,
    )

    return dbc.Card([
        dbc.CardHeader('Variables'),
        dbc.CardBody([
            select_all_none_variable_switch,
            dbc.Checklist(
                id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ID),
            ),
        ]),
    ])


def get_exploratory_analysis_method_radio():
    analysis_method_radio = dbc.RadioItems(
        inline=False,
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

    return dbc.Card([
        dbc.CardHeader('Analysis method'),
        dbc.CardBody(analysis_method_radio),
    ])


def get_exploratory_analysis_method_parameters_card():
    return dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody(
            dbc.Form([
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_1_ID)),
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_2_ID)),
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_3_ID)),
                dbc.Row(id=ddc.add_active_to_component_id(EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_4_ID)),
            ])
        ),
    ])


def get_exploratory_analysis_cardbody():
    return [
        dbc.Row(get_variables_checklist()),
        dbc.Row(get_exploratory_analysis_method_radio()),
        dbc.Row(get_exploratory_analysis_method_parameters_card()),
    ]


def get_exploratory_plot():
    graph = dcc.Graph(id=ddc.add_active_to_component_id(EXPLORATORY_GRAPH_ID)) # does it provide any performance improvement to scattergl?, config={'plotGlPixelRatio': 1})
    scatter_mode_radio = dbc.RadioItems(
        id=ddc.add_active_to_component_id(EXPLORATORY_GRAPH_SCATTER_MODE_RADIO_ID),
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
