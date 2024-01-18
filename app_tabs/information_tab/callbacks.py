from dash import Output, Input

from app_tabs.common.layout import APP_TABS_ID, SEARCH_DATASETS_TAB_VALUE
from app_tabs.information_tab.layout import PASS_INFO_BUTTON_ID
from utils.exception_handler import callback_with_exc_handling
from log import log_exception


@callback_with_exc_handling(
    Output(APP_TABS_ID, 'active_tab', allow_duplicate=True),
    Input(PASS_INFO_BUTTON_ID, 'n_clicks'),
    prevent_initial_call=True,
)
@log_exception
def foo_callback(n_clicks):
    return SEARCH_DATASETS_TAB_VALUE
