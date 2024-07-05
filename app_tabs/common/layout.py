from dash import dcc, html
import dash_bootstrap_components as dbc

import data_access


# Below there are id's of Dash JS components.
# The components themselves are declared in the dashboard layout (see the function get_dashboard_layout).
# Essential properties of each component are explained in the comments below.

APP_TABS_ID = 'app-tabs'    # see: https://dash.plotly.com/dash-core-components/tabs; method 1 (content as callback)
# value contains an id of the active tab
    # children contains a list of layouts of each tab

INFORMATION_TAB_VALUE = 'information-tab'
SEARCH_DATASETS_TAB_VALUE = 'search-datasets-tab'
SELECT_DATASETS_TAB_VALUE = 'select-datasets-tab'
FILTER_DATA_TAB_VALUE = 'filter-data-tab'
DATA_ANALYSIS_TAB_VALUE = 'data-analysis-tab'

SELECTED_STATIONS_STORE_ID = 'selected-stations-store'

SELECTED_ECV_STORE_ID = 'selected-ECV-store'

DATASETS_STORE_ID = 'datasets-store'
# 'data' stores datasets metadata in JSON, as provided by the method
# pd.DataFrame.to_json(orient='split', date_format='iso')

GANTT_SELECTED_DATASETS_IDX_STORE_ID = 'gantt-selected-datasets-idx-store'
GANTT_SELECTED_BARS_STORE_ID = 'gantt-selected-bars-store'

INTEGRATE_DATASETS_REQUEST_ID = 'integrate-datasets-request'
# 'data' stores a JSON representation of a request executed

FILTER_DATA_REQUEST_ID = 'filter-data-request'
# 'data' stores a JSON representation of a request executed

# https://community.plotly.com/t/dcc-graph-config-options/14672/2
GRAPH_CONFIG = {
    'autosizable': True,
    'displayModeBar': True,
    'doubleClick': 'autosize',
    #'fillFrame': True,
    #'editSelection': True,
    #'editable': False,
    'modeBarButtons': [['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d'], ['resetScale2d'], ['toImage']],
    'edits': {
        'titleText': True,
        'axisTitleText': True,
        'legendText': True,
        'colorbarTitleText': True,
        'annotationText': False,
        'annotationPosition': False,
    },
    'toImageButtonOptions': {
        'filename': 'atmo-access-plot',
        'format': 'png',
        'height': 800,
    },
    'showAxisDragHandles': True,
    'scrollZoom': True,
    'showAxisRangeEntryBoxes': True,
    'showTips': True,
    'displaylogo': False,
    'responsive': True,
}  # for more see: help(dcc.Graph)


NON_INTERACTIVE_GRAPH_CONFIG = {
    'autosizable': False,
    'displayModeBar': True,
    'editable': True,
    'edits': {
        'titleText': True,
        'axisTitleText': True,
        'legendText': True,
        'colorbarTitleText': True,
        'annotationText': False,
        'annotationPosition': False,
    },
    'modeBarButtons': [['autoScale2d'], ['toImage']],
    'toImageButtonOptions': {
        'filename': 'atmo-access-plot',
        'format': 'png',
        'height': 800,
    },
    'showAxisDragHandles': False,
    'showAxisRangeEntryBoxes': False,
    'showTips': True,
    'displaylogo': False,
    'responsive': True,
}  # for more see: help(dcc.Graph)


def get_app_data_stores():
    _all_variables = std_variables['value'].tolist()
    # these are special Dash components used for transferring data from one callback to other callback(s)
    # without displaying the data
    return [
        dcc.Store(id=SELECTED_STATIONS_STORE_ID, storage_type='session', data=None),
        dcc.Store(id=SELECTED_ECV_STORE_ID, storage_type='session', data=_all_variables),
        dcc.Store(id=DATASETS_STORE_ID, storage_type='session'),
        dcc.Store(id=INTEGRATE_DATASETS_REQUEST_ID, storage_type='session'),
        dcc.Store(id=FILTER_DATA_REQUEST_ID, storage_type='session'),
        dcc.Store(id=GANTT_SELECTED_DATASETS_IDX_STORE_ID, storage_type='session'),
        dcc.Store(id=GANTT_SELECTED_BARS_STORE_ID, storage_type='session'),
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


def get_help_icon(tag=''):
    return html.A(
        className='fa-solid fa-circle-info fa-2x',
        href=f'https://www.atmo-access.eu/atmo-access-time-series-analysis-service-help/{tag}',
        target='_blank',
    ),


def get_next_button(button_id):
    return dbc.Button(
        id=button_id,
        n_clicks=0,
        color='success',
        type='submit',
        children=html.Div(
            [
                html.Div('Next', style={'font-weight': 'bold', 'font-size': '135%'}),
                html.I(className='fa fa-arrow-circle-right fa-2x')
            ],
            style={
                'display': 'flex',
                'gap': '10px',
                'align-items': 'center'
            }
        ),
        #className='me-1',
        size='lg'
    )


def _get_std_variables(variables):
    std_vars = variables[['std_ECV_name', 'code']].drop_duplicates()
    # TODO: temporary
    try:
        std_vars = std_vars[std_vars['std_ECV_name'] != 'Aerosol Optical Properties']
    except ValueError:
        pass
    std_vars['label'] = std_vars['code'] + ' - ' + std_vars['std_ECV_name']
    return std_vars.rename(columns={'std_ECV_name': 'value'}).drop(columns='code')


variables = data_access.get_vars()
std_variables = _get_std_variables(variables)
