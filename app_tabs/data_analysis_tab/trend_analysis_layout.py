import pandas as pd
import dash_bootstrap_components as dbc
from dash import dcc

from utils import dash_dynamic_components as ddc, dash_persistence
from utils.graph_with_horizontal_selection_AIO import GraphWithHorizontalSelectionAIO
from . import common_layout
from app_tabs.common.layout import GRAPH_CONFIG


TREND_ANALYSIS_METHOD_RADIO_ID = 'trend-analysis-method-radio'
TREND_GRAPH_ID = 'trend-analysis-graph'
TREND_SUMMARY_CONTAINER_ID = 'trend-summary-container'

LINEAR_FIT_METHOD = 'Linear fit method'
NON_PARAMETRIC_MANN_KENDALL = 'Non-parametric Mann-Kendall'
THEIL_SEN_SLOPE_ESTIMATE = 'Theil-Sen slope estimate'
TREND_ANALYSIS_METHODS = [
    LINEAR_FIT_METHOD,
    NON_PARAMETRIC_MANN_KENDALL,
    THEIL_SEN_SLOPE_ESTIMATE,
]

DESEASONIZE_CHECKBOX_ID = 'trend-analysis-deseasonize-checkbox'

AGGREGATION_PERIOD_SELECT_ID = 'trend-analysis-aggregation-period-select'
NO_AGGREGATION = '-'
AGGREGATION_PERIOD_WORDINGS = {
    NO_AGGREGATION: ('no aggregation', None),
    'D': ('day', 'daily'),
    'W': ('week', 'weekly'),
    'M': ('month', 'monthly'),
    'Q': ('season', 'seasonal'),
    'Y': ('year', 'yearly'),
}
AGGREGATION_PERIOD_TIMEDELTA = {
    NO_AGGREGATION: None,
    'D': pd.Timedelta(1, 'D').to_timedelta64(),
    'W': pd.Timedelta(7, 'D').to_timedelta64(),
    'M': pd.Timedelta(31, 'D').to_timedelta64(),
    'Q': pd.Timedelta(92, 'D').to_timedelta64(),
    'Y': pd.Timedelta(365, 'D').to_timedelta64(),
}
DEFAULT_AGGREGATION_PERIOD = '-'

AGGREGATION_FUNCTION_SELECT_ID = 'trend-analysis-aggregation-function-select'
AGGREGATION_FUNCTIONS = {
    'mean': lambda resampled_series: resampled_series.mean(),
    'median': lambda resampled_series: resampled_series.median(),
}
DEFAULT_AGGREGATION_FUNCTION = 'mean'

TREND_ANALYSIS_AIO_ID = 'trend_analysis_time_filter'
TREND_ANALYSIS_AIO_CLASS = 'trend_analysis'


def get_time_filter():
    time_filter = GraphWithHorizontalSelectionAIO(
        aio_id=TREND_ANALYSIS_AIO_ID,
        aio_class=TREND_ANALYSIS_AIO_CLASS,
        # dynamic_component=True,
        x_axis_type='time',
        variable_label='time',
        x_label='time',
    )
    return time_filter


def get_trend_analysis_cardbody(time_filter):
    analysis_method_radio = dbc.RadioItems(
        id=ddc.add_active_to_component_id(TREND_ANALYSIS_METHOD_RADIO_ID),
        options=[
            {'label': analysis_method, 'value': analysis_method}
            for analysis_method in TREND_ANALYSIS_METHODS
        ],
        value=TREND_ANALYSIS_METHODS[0],
        inline=False,
        persistence=True,
        persistence_type='session',
    ),

    trend_analysis_method_radio = dbc.Card([
        dbc.CardHeader('Analysis method'),
        dbc.CardBody(analysis_method_radio),
    ])

    deseasonize_checkbox = dbc.Checkbox(
        id=ddc.add_active_to_component_id(DESEASONIZE_CHECKBOX_ID),
        label='De-seasonise',
        value=False,
    )

    aggregation_period_select = [
        dbc.Col(dbc.Label('Aggregation period:'), width=6),
        dbc.Col(
            dbc.Select(
                id=ddc.add_active_to_component_id(AGGREGATION_PERIOD_SELECT_ID),
                options=[
                    {'label': period_label, 'value': period_id}
                    for period_id, (period_label, _) in AGGREGATION_PERIOD_WORDINGS.items()
                ],
                value=DEFAULT_AGGREGATION_PERIOD,
                **dash_persistence.get_dash_persistence_kwargs(True)
            ),
            width=6
        )
    ]

    aggregation_function_select = [
        dbc.Col(dbc.Label('Aggregation function:'), width=6),
        dbc.Col(
            dbc.Select(
                id=ddc.add_active_to_component_id(AGGREGATION_FUNCTION_SELECT_ID),
                options=[
                    {'label': agg_func_label, 'value': agg_func_label}
                    for agg_func_label in AGGREGATION_FUNCTIONS.keys()
                ],
                value=DEFAULT_AGGREGATION_FUNCTION,
                **dash_persistence.get_dash_persistence_kwargs(True)
            ),
            width=6
        )
    ]

    trend_analysis_method_parameters_card = dbc.Card([
        dbc.CardHeader('Parameters'),
        dbc.CardBody([
            dbc.Form([
                dbc.Row(deseasonize_checkbox),
                dbc.Row(aggregation_period_select),
                dbc.Row(aggregation_function_select),
                dbc.Row(common_layout.minimal_sample_size_input),
                dbc.Row([
                    dbc.Label('Time filter:'),
                    time_filter.get_range_controller(),
                ])
            ])
        ]),
    ])

    return [
        time_filter.get_data_stores(),
        dbc.Row(common_layout.variables_checklist),
        dbc.Row(trend_analysis_method_radio),
        dbc.Row(trend_analysis_method_parameters_card),
    ]


def get_trend_summary_container():
    return dbc.Container(id=ddc.add_active_to_component_id(TREND_SUMMARY_CONTAINER_ID))
