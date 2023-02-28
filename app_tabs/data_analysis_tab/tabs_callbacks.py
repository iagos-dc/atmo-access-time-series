from dash import Input, callback, Output

from app_tabs.data_analysis_tab.tabs_layout import DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, DATA_ANALYSIS_FIGURE_CONTAINER_ID, \
    KIND_OF_ANALYSIS_TABS_ID, EXPLORATORY_ANALYSIS_TAB_ID, TREND_ANALYSIS_TAB_ID, MULTIVARIATE_ANALYSIS_TAB_ID
from app_tabs.data_analysis_tab.exploratory_analysis_layout import exploratory_analysis_cardbody, \
    exploratory_plot
from app_tabs.data_analysis_tab.trend_analysis_layout import trend_plot, trend_summary_container
from app_tabs.data_analysis_tab.multivariate_analysis_layout import multivariate_analysis_cardbody, multivariate_plot
from log import log_exception


@callback(
    Output(DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, 'children'),
    Output(DATA_ANALYSIS_FIGURE_CONTAINER_ID, 'children'),
    Input(KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
)
@log_exception
def get_data_analysis_carbody_content(tab_id):
    if tab_id == EXPLORATORY_ANALYSIS_TAB_ID:
        param_cardbody_children = exploratory_analysis_cardbody
        figure_container_children = exploratory_plot
    elif tab_id == TREND_ANALYSIS_TAB_ID:
        param_cardbody_children = trend_summary_container
        figure_container_children = trend_plot
    elif tab_id == MULTIVARIATE_ANALYSIS_TAB_ID:
        param_cardbody_children = multivariate_analysis_cardbody
        figure_container_children = multivariate_plot
    else:
        raise ValueError(f'unknown tab_id={tab_id}')
    return param_cardbody_children, figure_container_children
