from dash import dcc
import dash_bootstrap_components as dbc


# Below there are id's of Dash JS components.
# The components themselves are declared in the dashboard layout (see the function get_dashboard_layout).
# Essential properties of each component are explained in the comments below.

APP_TABS_ID = 'app-tabs'    # see: https://dash.plotly.com/dash-core-components/tabs; method 1 (content as callback)
# value contains an id of the active tab
    # children contains a list of layouts of each tab

DATASETS_STORE_ID = 'datasets-store'
# 'data' stores datasets metadata in JSON, as provided by the method
# pd.DataFrame.to_json(orient='split', date_format='iso')

INTEGRATE_DATASETS_REQUEST_ID = 'integrate-datasets-request'
# 'data' stores a JSON representation of a request executed

FILTER_DATA_REQUEST_ID = 'filter-data-request'
# 'data' stores a JSON representation of a request executed

GRAPH_CONFIG = {
    'autosizable': True,
    'displayModeBar': True,
    'doubleClick': 'autosize',
    #'fillFrame': True,
    #'editSelection': True,
    #'editable': False,
    'modeBarButtons': [['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d'], ['autoScale2d'], ['toImage']],
    'edits': {
        'titleText': True,
        'axisTitleText': True,
        'legendText': True,
        'colorbarTitleText': True,
    },
    'toImageButtonOptions': {
        'filename': 'foo',
        'format': 'png',
        'height': 800,
    },
    'showAxisDragHandles': True,
    'showAxisRangeEntryBoxes': True,
    'showTips': True,
    'displaylogo': False,
    # 'responsive': True,
}  # for more see: help(dcc.Graph)


NON_INTERACTIVE_GRAPH_CONFIG = {
    'autosizable': False,
    'displayModeBar': True,
    'editable': False,
    'modeBarButtons': [['toImage']],
    'toImageButtonOptions': {
        'filename': 'foo',
        'format': 'png',
        'height': 800,
    },
    'showAxisDragHandles': False,
    'showAxisRangeEntryBoxes': False,
    'showTips': True,
    'displaylogo': False,
    # 'responsive': True,
}  # for more see: help(dcc.Graph)




def get_app_data_stores():
    # these are special Dash components used for transferring data from one callback to other callback(s)
    # without displaying the data
    return [
        dcc.Store(id=DATASETS_STORE_ID, storage_type='session'),
        dcc.Store(id=INTEGRATE_DATASETS_REQUEST_ID, storage_type='session'),
        dcc.Store(id=FILTER_DATA_REQUEST_ID, storage_type='session'),
    ]


def _tooltip_target_to_str(target):
    if isinstance(target, dict):
        target_as_str = '_'.join(f'{k}-{v}' for k, v in target.items())
    elif isinstance(target, str):
        target_as_str = target
    else:
        raise TypeError(f'target must be either str or dict; got type(target)={type(target)}')
    return f'tooltip-to-{target_as_str}'


def get_tooltip(tooltip_text, target, **kwargs):
    tooltip_kwargs = {
        'placement': 'top-start',
        'style': {'font-size': '0.8em'}
    }
    tooltip_kwargs.update(kwargs)

    return dbc.Tooltip(
        tooltip_text,
        id=_tooltip_target_to_str(target),
        target=target,
        **tooltip_kwargs
    )
