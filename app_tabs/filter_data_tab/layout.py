from dash import html, dcc
import dash_bootstrap_components as dbc


FILTER_TAB_CONTAINER_ROW_ID = 'filter-tab-container-row'
    # 'children' contains a layout of the filter tab
FILTER_TYPE_RADIO_ID = 'filter_type_radio'
FILTER_TIME_CONINCIDENCE_SELECT_ID = 'filter_time_coincidence_select'


def get_log_axis_switches(i):
    return dbc.Checklist(
        options=[
            {'label': 'x-axis in log-scale', 'value': 'log_x'},
            {'label': 'y-axis in log-scale', 'value': 'log_y'},
        ],
        value=[],
        id={'subcomponent': 'log_scale_switch', 'aio_id': i},
        inline=True,
        switch=True,
    )


def get_nbars_slider(i):
    emptry_row = dbc.Row(dbc.Col(html.P()))
    row = dbc.Row(
        [
            dbc.Col(dbc.Label('Number of histogram bars:'), width='auto'),
            dbc.Col(
                dbc.RadioItems(
                    options=[{'label': str(nbars), 'value': nbars} for nbars in [10, 20, 30, 50, 100]],
                    value=50,
                    inline=True,
                    id={'subcomponent': 'nbars_slider', 'aio_id': i},
                ),
                width='auto',
            ),
        ],
        justify='end', #align='baseline',
    )
    return [emptry_row, row]


def get_time_granularity_radio():
    return dbc.InputGroup([
        dbc.InputGroupText('View by: '),
        dbc.RadioItems(
            options=[
                {"label": "year", "value": 'year'},
                {"label": "season", "value": 'season'},
                {"label": "month", "value": 'month'},
            ],
            value='year',
            id={'subcomponent': 'time_granularity_radio', 'aio_id': 'time_filter-time'},
            inline=True,
        ),
    ])


def get_filtering_type_radio():
    simple_vs_cross_filter_radio = dbc.RadioItems(
        options=[
            {'label': 'Simple filter', 'value': 'simple filter'},
            {'label': 'Cross filter', 'value': 'cross filter'},
        ],
        value='simple filter',
        inline=True,
        id=FILTER_TYPE_RADIO_ID,
    )

    time_coincidence_select = dbc.InputGroup(
        [
            dbc.InputGroupText('Observations coincidence time'),
            dbc.Select(
                options=[
                    {'label': '1 hour', 'value': '1H'},
                    {'label': '3 hour', 'value': '3H'},
                    {'label': '6 hour', 'value': '6H'},
                    {'label': '12 hour', 'value': '12H'},
                    {'label': '24 hour', 'value': '24H'},
                    {'label': '48 hour', 'value': '48H'},
                    {'label': '72 hour', 'value': '72H'},
                    {'label': '7 days', 'value': '7D'},
                    {'label': '14 days', 'value': '14D'},
                    {'label': '30 days', 'value': '30D'},
                    {'label': 'custom', 'value': 'custom'},
                ],
                value='24H',
                disabled=True,
                style={'background-color': '#dddddd'},
                id=FILTER_TIME_CONINCIDENCE_SELECT_ID,
            ),
        ],
        id='filter_time_coincidence_select-time_filter-time-tooltip_target',
    )

    tooltip = dbc.Tooltip(
        'Observations of different variables will be considered as coinciding in time if the difference between time measurements does not exceed the selected value',
        target='filter_time_coincidence_select-time_filter-time-tooltip_target',
    )

    cols = [
        dbc.Col(simple_vs_cross_filter_radio, width='auto'),
        dbc.Col([time_coincidence_select, tooltip], width='auto'),
    ]
    return cols


FILTER_DATA_TAB_VALUE = 'filter-data-tab'
FILTER_DATA_BUTTON_ID = 'filter-data-button'


def get_filter_data_tab():
    return dcc.Tab(
        label='Filter data',
        value=FILTER_DATA_TAB_VALUE,
        children=html.Div(
            style={'margin': '20px'},
            children=dbc.Container(
                [
                    dbc.Row(
                        get_filtering_type_radio() +
                        [
                            dbc.Col(
                                dbc.Button(
                                    id=FILTER_DATA_BUTTON_ID, n_clicks=0,
                                    color='primary',
                                    type='submit',
                                    style={'font-weight': 'bold'},
                                    children='Apply filter to data'
                                ),
                                width='auto',
                            ),
                        ],
                        justify='start',
                    ),
                    dbc.Row(id=FILTER_TAB_CONTAINER_ROW_ID),
                ],
                fluid=True,
            )
        )
    )