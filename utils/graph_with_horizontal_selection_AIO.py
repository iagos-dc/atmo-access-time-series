import dash
from dash import dcc, html, callback, MATCH, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from dash.development.base_component import Component


from log import start_logging_callbacks, log_callback_with_ret_value


start_logging_callbacks('/home/wolp/PycharmProjects/atmo-access-time-series/log/log.mmap')


def _my_explicitize_args(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not Component.UNDEFINED}


def _component_id(component_name):
    def _(aio_id):
        return {
            'component': 'GraphWithHorizontalSelectionAIO',
            'subcomponent': component_name,
            'aio_id': aio_id,
        }

    return _


selected_range_store_id = _component_id('selected_range_store')
figure_data_store_id = _component_id('figure_data_store')
graph_id = _component_id('graph')
from_input_id = _component_id('from_input')
to_input_id = _component_id('to_input')
reset_selection_button_id = _component_id('reset_selection_button')
foo_container_id = _component_id('foo_asd')


@callback(
    Output(graph_id(MATCH), 'figure'),
    Input(figure_data_store_id(MATCH), 'data'),
    Input(selected_range_store_id(MATCH), 'data'),
    State(graph_id(MATCH), 'figure'),
)
@log_callback_with_ret_value()
def update_graph_figure(fig_data, selected_range, fig):
    if fig is None:
        fig = go.Figure()

    if fig_data is not None:
        fig['data'] = fig_data['fig']

    fig['layout'].pop('shapes', None)
    fig['layout'].pop('selections', None)
    x0, x1 = selected_range if selected_range is not None else (None, None)
    if x0 is None:
        x0 = fig_data['rng'][0]
    if x1 is None:
        x1 = fig_data['rng'][1]
    if True or (x0 is not None or x1 is not None):
        fig['layout']['shapes'] = [
            {
                'fillcolor': 'blue',
                'opacity': 0.1,
                'type': 'rect',
                'x0': x0,
                'x1': x1,
                'xref': 'x',
                'y0': 0,
                'y1': 1,
                'yref': 'y domain'
            }
        ]
    return fig


@callback(
    Output(from_input_id(MATCH), 'value'),
    Output(to_input_id(MATCH), 'value'),
    Output(from_input_id(MATCH), 'invalid'),
    Output(to_input_id(MATCH), 'invalid'),
    Output(selected_range_store_id(MATCH), 'data'),
    Input(graph_id(MATCH), 'selectedData'),
    Input(reset_selection_button_id(MATCH), 'n_clicks'),
    Input(from_input_id(MATCH), 'value'),
    Input(to_input_id(MATCH), 'value'),
    State(selected_range_store_id(MATCH), 'data'),
)
@log_callback_with_ret_value()
def update_from_and_to_input_values(selected_data_on_fig, reset_selection_n_clicks, x0, x1, selected_range):
    if ctx.triggered_id is None:
        return None, None, False, False, [None, None]

    x_axis_type = ctx.outputs_list[0]['id']['aio_id'].split('-')[-1]
    if ctx.triggered_id.subcomponent == 'reset_selection_button':
        return None, None, False, False, [None, None]
    elif ctx.triggered_id.subcomponent == 'graph':
        if selected_data_on_fig is not None and 'range' in selected_data_on_fig:
            new_x0, new_x1 = selected_data_on_fig['range']['x']
            if x_axis_type == 'time':
                new_x0 = pd.Timestamp(new_x0).strftime('%Y-%m-%d %H:%M')
                new_x1 = pd.Timestamp(new_x1).strftime('%Y-%m-%d %H:%M')
            return new_x0, new_x1, False, False, update_selected_range(new_x0, new_x1, True, True, selected_range)
        else:
            raise PreventUpdate
    elif ctx.triggered_id.subcomponent in ['from_input', 'to_input']:
        if x_axis_type == 'time':
            def get_x(x, x_valid):
                if x_valid is None:
                    return None
                elif x_valid:
                    return pd.Timestamp(x).strftime('%Y-%m-%d %H:%M')
                else:
                    return dash.no_update

            x0_valid = _valid(x0)
            x1_valid = _valid(x1)

            x0 = get_x(x0, x0_valid)
            x1 = get_x(x1, x1_valid)
        else:
            x0_valid = True
            x1_valid = True

        def is_invalid(valid):
            if valid is False:
                return True
            else:
                return False

        return x0, x1, is_invalid(x0_valid), is_invalid(x1_valid), update_selected_range(x0, x1, x0_valid, x1_valid, selected_range)
    else:
        raise RuntimeError(f'unknown callback trigger: ctx.triggered_id={ctx.triggered_id}')


def _valid(x):
    try:
        x = pd.Timestamp(x)
    except ValueError:
        return False
    if pd.isnull(x):
        return None
    return True


def update_selected_range(x0, x1, x0_valid, x1_valid, selected_range):
    if selected_range is not None:
        old_x0, old_x1 = selected_range
    else:
        old_x0, old_x1 = None, None

    if x0_valid is None:
        x0 = None
    elif x0_valid is False:
        x0 = old_x0

    if x1_valid is None:
        x1 = None
    elif x1_valid is False:
        x1 = old_x1

    return [x0, x1]


def figure_update_layout(figure):
    figure.update_layout(
        dragmode='select',
        selectdirection='h',
        yaxis={'fixedrange': True}
        # activeselection={'fillcolor': 'yellow'},
    )
    return figure


def graph(aio_id, figure=Component.UNDEFINED):
    if figure is not Component.UNDEFINED:
        figure = figure_update_layout(figure)

    return dcc.Graph(
        id=graph_id(aio_id),
        **_my_explicitize_args(figure=figure),
    )


def time_input_field(aio_id):
    placeholder = 'YYYY-MM-DD HH:MM'
    return dbc.Input(
        id=from_input_id(aio_id),
        type='text',
        debounce=True,
        placeholder=placeholder,
        maxLength=len(placeholder),
        invalid=False,
    )


def scalar_input_field(i):
    return dbc.Input(
        id=i,
        type='number',
        debounce=True,
        invalid=False,
    )


def interval_inputs(aio_id, x_axis_type, x_label=None):
    if x_axis_type == 'time':
        return dbc.InputGroup(
            [
                dbc.InputGroupText('From'),
                time_input_field(from_input_id(aio_id)),
                dbc.InputGroupText('to'),
                time_input_field(to_input_id(aio_id)),
            ],
            className='mb-3',
        )
    else:
        return dbc.InputGroup(
            [
                scalar_input_field(from_input_id(aio_id)),
                dbc.InputGroupText(f'< {x_label if x_label is not None else "x"} <'),
                scalar_input_field(to_input_id(aio_id)),
            ],
            className='mb-3',
        )



def interval_controls_container(aio_id, x_axis_type, x_label=None):
    return [
        dbc.ListGroup(
            [
                interval_inputs(aio_id, x_axis_type, x_label=x_label), #, 0),
            ],
        ),
        #dbc.Row(
        #    dbc.Col(
        #        dbc.Button(
        #            #children='Add new interval',
        #            html.I(className='fa-solid fa-plus'),
        #            id=self.ids.new_interval_button(aio_ids),
        #            n_clicks=0,
        #            color='primary', type='submit', style={'font-weight': 'bold'},
        #        ),
        #        width='auto'
        #    ),
        #    justify='end',
        #),
        dbc.Button(
            children='Reset selection',
            id=reset_selection_button_id(aio_id)
        ),
    ]


def selected_range_store(aio_id, data=None):
    return dcc.Store(id=selected_range_store_id(aio_id), data=data)


def figure_data_store(aio_id, data=None):
    return dcc.Store(id=figure_data_store_id(aio_id), data=data)


class GraphWithHorizontalSelectionAIO(html.Div):
    def __init__(self, aio_id, x_axis_type, x_min=None, x_max=None, x_label=None, title=None, figure=Component.UNDEFINED, **kwargs):
        if x_axis_type not in ['time', 'scalar']:
            raise ValueError(f"x_axis_type must be 'time' or 'scalar'; got {x_axis_type}")

        aio_id = aio_id + '-' + x_axis_type
        figure_data = None
        if figure is not Component.UNDEFINED:
            figure_data = {
                'fig': figure['data'],
                'rng': [x_min, x_max],
            }
        super().__init__(
            children=[
                selected_range_store(aio_id, [None, None]),
                figure_data_store(aio_id, figure_data),
                dbc.Container(
                    [
                        #html.H1('GraphWithHorizontalSelectionAIO'),
                        #html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardHeader(title if title is not None else 'Interval selected:'),
                                        dbc.CardBody(
                                            children=interval_controls_container(aio_id, x_axis_type, x_label=x_label),
                                        ),
                                    ])
                                ),
                                dbc.Col(graph(aio_id, figure=figure), width=8),
                            ],
                            align='start',
                        ),
                    ],
                    fluid=True,
                )
            ],
            #className='row',
            **kwargs
        )
