from dash import Input, callback, Output
import dash_bootstrap_components as dbc

from app_tabs.common.layout import FILTER_DATA_REQUEST_ID, DATA_ANALYSIS_TAB_VALUE
from app_tabs.data_analysis_tab.tabs_layout import DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, DATA_ANALYSIS_FIGURE_CONTAINER_ID, \
    KIND_OF_ANALYSIS_TABS_ID, EXPLORATORY_ANALYSIS_TAB_ID, TREND_ANALYSIS_TAB_ID, MULTIVARIATE_ANALYSIS_TAB_ID, BORDER_STYLE
from app_tabs.data_analysis_tab.exploratory_analysis_layout import exploratory_analysis_cardbody, \
    exploratory_plot
from app_tabs.data_analysis_tab.trend_analysis_layout import get_time_filter, get_trend_analysis_cardbody, \
    trend_graph, align_y_axes_button, autocorrelation_graph, trend_summary_bar_graph
from app_tabs.data_analysis_tab.multivariate_analysis_layout import multivariate_analysis_cardbody, multivariate_plot
from utils.exception_handler import callback_with_exc_handling, AppException


@callback_with_exc_handling(
    Output(DATA_ANALYSIS_TAB_VALUE, 'disabled'),
    Input(FILTER_DATA_REQUEST_ID, 'data'),
)
def enable_data_analysis_tab(filter_data_request):
    return filter_data_request is None


@callback_with_exc_handling(
    Output(DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, 'children'),
    Output(DATA_ANALYSIS_FIGURE_CONTAINER_ID, 'children'),
    Input(KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
)
def get_data_analysis_carbody_content(tab_id):
    if tab_id == EXPLORATORY_ANALYSIS_TAB_ID:
        param_cardbody_children = exploratory_analysis_cardbody
        figure_container_children = exploratory_plot
    elif tab_id == TREND_ANALYSIS_TAB_ID:
        time_filter = get_time_filter()
        param_cardbody_children = get_trend_analysis_cardbody(time_filter)
        figure_container_children = dbc.Container(
            [
                dbc.Row(
                    dbc.Col([trend_graph, align_y_axes_button], width=12, style=BORDER_STYLE),
                ),
                dbc.Row([
                    dbc.Col(trend_summary_bar_graph, width=6, style=BORDER_STYLE),
                    dbc.Col(autocorrelation_graph, width=6, style=BORDER_STYLE),
                ])
            ],
            fluid=True
        )
    elif tab_id == MULTIVARIATE_ANALYSIS_TAB_ID:
        param_cardbody_children = multivariate_analysis_cardbody
        figure_container_children = multivariate_plot
    else:
        raise ValueError(f'unknown tab_id={tab_id}')
    return param_cardbody_children, figure_container_children
