import dash_bootstrap_components as dbc
from dash import dcc

from utils import dash_dynamic_components as ddc, dash_persistence
from app_tabs.common import layout as common_layout


TREND_GRAPH_ID = 'trend-analysis-graph'
TREND_SUMMARY_CONTAINER_ID = 'trend-summary-container'


def _get_trend_plot():
    graph = dcc.Graph(
        id=ddc.add_active_to_component_id(TREND_GRAPH_ID),
        config=common_layout.GRAPH_CONFIG,
        # responsive=True,  # WARNING: this triggers relayoutData={'autosize': True}
    )

    return graph


trend_plot = _get_trend_plot()


trend_summary_container = dbc.Container(
    id=ddc.add_active_to_component_id(TREND_SUMMARY_CONTAINER_ID),
)
