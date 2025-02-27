import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc, html

from utils import dash_dynamic_components as ddc, dash_persistence
from app_tabs.common import layout as common_layout


MULTIVARIATE_ANALYSIS_VARIABLES_CARDHEADER_ID = 'multivariate-analysis-variables-cardheader'
MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ID = 'multivariate-analysis-variables-cardbody'
MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_2_ID = 'multivariate-analysis-parameters-form-row-2'
MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_3_ID = 'multivariate-analysis-parameters-form-row-3'
PLOT_TYPE_RADIO_ID = 'plot-type-radio'
INDIVIDUAL_OBSERVATIONS_PLOT = 'individual-observations-plot'
HEXBIN_PLOT = 'hexbin-plot'

MULTIVARIATE_GRAPH_ID = 'data-analysis-multivariate-graph'

X_VARIABLE_SELECT_ID = 'x-variable-select'
Y_VARIABLE_SELECT_ID = 'y-variable-select'
C_VARIABLE_SELECT_ID = 'c-variable-select'

HEXBIN_PLOT_RESOLUTION_SLIDER_ID = 'hexbin-plot-resolution-slider'

AGGREGATORS_CHECKLIST_ID = 'aggregators-checklist-id'
AGGREGATOR_DISPLAY_BUTTONS_FORM_ID = 'aggregator-display-buttons-form'
AGGREGATOR_DISPLAY_BUTTONS_ID = 'aggregator_display_buttons'

MIN_AGG = 'min'
Q5_AGG = 'p5'
MEAN_MINUS_STD_AGG = 'mean-std'
MEAN_AGG = 'mean'
MEAN_PLUS_STD_AGG = 'mean+std'
Q95_AGG = 'p95'
MAX_AGG = 'max'
AGGREGATOR_FUNCTIONS = {
    MIN_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.min(a),
    Q5_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.quantile(a, q=0.05),
    MEAN_MINUS_STD_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.mean(a) - pd.core.groupby.DataFrameGroupBy.std(a, ddof=0),
    MEAN_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.mean(a),
    MEAN_PLUS_STD_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.mean(a) + pd.core.groupby.DataFrameGroupBy.std(a, ddof=0),
    Q95_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.quantile(a, q=0.95),
    MAX_AGG: lambda a: pd.core.groupby.DataFrameGroupBy.max(a),
}
AGGREGATOR_RADIOITEMS_OPTIONS = list(AGGREGATOR_FUNCTIONS)


def _get_multivariate_analysis_cardbody():
    variables_card = dbc.Card([
        dbc.CardHeader(
            children='Variables',
            id=ddc.add_active_to_component_id(MULTIVARIATE_ANALYSIS_VARIABLES_CARDHEADER_ID),
        ),
        dbc.CardBody(
            id=ddc.add_active_to_component_id(MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ID)
        ),
    ])

    plot_type = [
        dbc.Label('Plot type:', width='auto'),
        dbc.Col(
            dbc.RadioItems(
                id=ddc.add_active_to_component_id(PLOT_TYPE_RADIO_ID),
                options=[
                    {'label': 'hexagonal bins', 'value': HEXBIN_PLOT},
                    {'label': 'individual observations', 'value': INDIVIDUAL_OBSERVATIONS_PLOT},
                ],
                value=HEXBIN_PLOT,
                inline=True,
                persistence=True,
                persistence_type='session',
            ),
            width='auto',
        )
    ]
    parameters_card = dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody(
            dbc.Form([
                dbc.Row(plot_type),
                dbc.Row(id=ddc.add_active_to_component_id(MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_2_ID)),
                dbc.Row(id=ddc.add_active_to_component_id(MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_3_ID)),
            ]),
        ),
    ])

    return [
        dbc.Row(variables_card),
        dbc.Row(parameters_card),
    ]


multivariate_analysis_cardbody = _get_multivariate_analysis_cardbody()


def _get_multivariate_plot():
    graph = dcc.Graph(
        id=ddc.add_active_to_component_id(MULTIVARIATE_GRAPH_ID),
        config=common_layout.GRAPH_CONFIG,
        # responsive=True,  # WARNING: this triggers relayoutData={'autosize': True}
    )  # does it provide any performance improvement to scattergl?, config={'plotGlPixelRatio': 1})
    return dbc.Row(graph)


multivariate_plot = _get_multivariate_plot()


def get_variable_dropdown(dropdown_id, axis_label, options, value, disabled=False, persistence_id=None):
    return dbc.InputGroup([
        dbc.InputGroupText(axis_label),
        dbc.Select(
            id=dropdown_id,
            options=options,
            value=value,
            disabled=disabled,
            **dash_persistence.get_dash_persistence_kwargs(persistence_id=persistence_id),
        ),
    ])


hexbin_plot_resolution_slider = dbc.Form(
    dbc.Row([
        dbc.Label('Hex-bin plot resolution', width=4),
        dbc.Col(
            dcc.Slider(
                id=ddc.add_active_to_component_id(HEXBIN_PLOT_RESOLUTION_SLIDER_ID),
                min=10, max=60, step=10, value=30,
                **dash_persistence.get_dash_persistence_kwargs(persistence_id=True)
            ),
            width=8
        )
    ]),
)


def _get_choice_of_aggregators():
    aggregators_radioitems = dbc.RadioItems(
        id=ddc.add_active_to_component_id(AGGREGATOR_DISPLAY_BUTTONS_ID),
        options=[
            {'label': agg_option, 'value': agg_option}
            for agg_option in AGGREGATOR_RADIOITEMS_OPTIONS
        ],
        value=MEAN_AGG,
        inline=True,
    )

    return dbc.Form([
        dbc.Label(f'Aggregate C-variable in hex-bin using'),
        aggregators_radioitems,
    ])


choice_of_aggregators = _get_choice_of_aggregators()
