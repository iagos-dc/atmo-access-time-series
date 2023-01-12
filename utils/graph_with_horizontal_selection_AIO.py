import pkg_resources
import dash
from dash import dcc, html, callback, MATCH, ctx, ALL
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from dash.development.base_component import Component

from log import log_exception
from log import start_logging_callbacks, log_callback_with_ret_value


# log_filepath = pkg_resources.resource_filename('log', 'log_callbacks.pkl')
# start_logging_callbacks(log_filepath)


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
@log_exception
#@log_callback_with_ret_value()
def update_graph_figure(fig_data, selected_range, fig):
    # TODO: since fig is now a figure, apply update_layout methods properly!
    if fig is None:
        fig = go.Figure()

    if fig_data is not None:
        fig = figure_update_layout(go.Figure(fig_data['fig']))

    fig['layout'].pop('shapes', None)
    fig['layout'].pop('selections', None)

    if selected_range is not None:
        variable_label, x0, x1 = selected_range['variable_label'], selected_range['x_sel_min'], selected_range['x_sel_max']
    else:
        raise RuntimeError('we should not be here!!!')

    if x0 is None and fig_data is not None:
        x0 = fig_data['rng'][0]
    if x1 is None and fig_data is not None:
        x1 = fig_data['rng'][1]
    if True or (x0 is not None or x1 is not None):
        fig['layout']['shapes'] = [
            {
                'line': {'color': 'red', 'width': 2},
                #'fillcolor': 'blue',
                #'opacity': 0.1,
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
@log_exception
# @log_callback_with_ret_value()
def update_from_and_to_input_values(selected_data_on_fig, reset_selection_n_clicks, x0, x1, selected_range):
    if ctx.triggered_id is None:
        raise dash.exceptions.PreventUpdate
        # return None, None, False, False, {'variable_label': selected_range['variable_label'], 'x_sel_min': None, 'x_sel_max': None}

    x_axis_type = ctx.outputs_list[0]['id']['aio_id'].split('-')[-1]
    if ctx.triggered_id.subcomponent == 'reset_selection_button':
        return None, None, False, False, {'variable_label': selected_range['variable_label'], 'x_sel_min': None, 'x_sel_max': None}
    elif ctx.triggered_id.subcomponent == 'graph':
        if selected_data_on_fig is not None and 'range' in selected_data_on_fig:
            new_x0, new_x1 = selected_data_on_fig['range']['x']

            # plotly bug??? selected_data_on_fig['range']['x'] might have new_x0 > new_x1; so need to fix it:
            if new_x0 > new_x1:
                new_x0, new_x1 = new_x1, new_x0

            if x_axis_type == 'time':
                new_x0 = pd.Timestamp(new_x0).strftime('%Y-%m-%d %H:%M')
                new_x1 = pd.Timestamp(new_x1).strftime('%Y-%m-%d %H:%M')
            return new_x0, new_x1, False, False, update_selected_range(new_x0, new_x1, True, True, selected_range)
        else:
            return x0, x1, False, False, update_selected_range(x0, x1, True, True, selected_range)
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

        update_sel_range = update_selected_range(x0, x1, x0_valid, x1_valid, selected_range)
        return x0, x1, is_invalid(x0_valid), is_invalid(x1_valid), update_sel_range
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
        variable_label, old_x0, old_x1 = selected_range['variable_label'], selected_range['x_sel_min'], selected_range['x_sel_max']
    else:
        raise RuntimeError('we should not be here!')
        # variable_label, old_x0, old_x1 = None, None, None

    if x0_valid is None:
        x0 = None
    elif x0_valid is False:
        x0 = old_x0

    if x1_valid is None:
        x1 = None
    elif x1_valid is False:
        x1 = old_x1

    return {'variable_label': variable_label, 'x_sel_min': x0, 'x_sel_max': x1}


def figure_update_layout(figure):
    figure.update_layout(
        dragmode='select',
        selectdirection='h',
        yaxis={'fixedrange': True}
        # activeselection={'fillcolor': 'yellow'},
    )
    return figure


def figure_dict_update_layout(figure_dict):
    if figure_dict is None:
        return None
    try:
        layout = figure_dict['layout']
    except KeyError:
        return figure_dict
    layout.update(
        dragmode='select',
        selectdirection='h',
        yaxis={'fixedrange': True}
    )
    return figure_dict


def graph(aio_id, figure=Component.UNDEFINED):
    if figure is not Component.UNDEFINED:
        figure = figure_update_layout(figure)

    return dcc.Graph(
        id=graph_id(aio_id),
        # config={'modeBarButtonsToAdd': ['select'], 'modeBarButtonsToRemove': ['zoom']},
        **_my_explicitize_args(figure=figure),
    )


def time_input_field(i):
    placeholder = 'YYYY-MM-DD HH:MM'
    return dbc.Input(
        id=i,
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


def interval_controls_container(aio_id, x_axis_type, x_label=None, extra_dash_components=None, extra_dash_components2=None):
    if extra_dash_components is None:
        extra_dash_components = []
    if not isinstance(extra_dash_components, (list, tuple)):
        extra_dash_components = [extra_dash_components]
    if extra_dash_components2 is None:
        extra_dash_components2 = []
    if not isinstance(extra_dash_components2, (list, tuple)):
        extra_dash_components2 = [extra_dash_components2]

    return [
        dbc.Row(dbc.Col(interval_inputs(aio_id, x_axis_type, x_label=x_label))),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        children='Reset selection',
                        id=reset_selection_button_id(aio_id)
                    ),
                    width='auto',
                    align='center',
                ),
            ] +
            [
                dbc.Col(extra_dash_component, width='auto', align='center') for extra_dash_component in extra_dash_components
            ],
            justify='between'
        ),
    ] + [
        extra_dash_component for extra_dash_component in extra_dash_components2
    ]


def selected_range_store(aio_id, variable_label=None, x_sel_min=None, x_sel_max=None):
    data = {'variable_label': variable_label, 'x_sel_min': x_sel_min, 'x_sel_max': x_sel_max}
    return dcc.Store(id=selected_range_store_id(aio_id), data=data)


# def all_selected_ranges_store(data=None):
#     return dcc.Store(id=all_selected_ranges_store_id, data=data)


def figure_data_store(aio_id, data=None):
    return dcc.Store(id=figure_data_store_id(aio_id), data=data)


class GraphWithHorizontalSelectionAIO(dbc.Container):
    def __init__(
            self,
            aio_id,
            x_axis_type,
            variable_label=None,
            x_min=None,
            x_max=None,
            x_label=None,
            title=None,
            figure=Component.UNDEFINED,
            extra_dash_components=None,
            extra_dash_components2=None,
            **kwargs
    ):
        if x_axis_type not in ['time', 'scalar']:
            raise ValueError(f"x_axis_type must be 'time' or 'scalar'; got {x_axis_type}")

        aio_id = aio_id + '-' + x_axis_type
        figure_data = None
        if figure is not Component.UNDEFINED:
            figure_data = {
                'fig': figure,
                'rng': [x_min, x_max],
            }

        children = [
           selected_range_store(
               aio_id,
               variable_label=variable_label,
               x_sel_min=None,
               x_sel_max=None,
           ),
           figure_data_store(aio_id, figure_data),
           dbc.Container(
               [
                   # html.H1('GraphWithHorizontalSelectionAIO'),
                   # html.Hr(),
                   dbc.Row(
                       [
                           dbc.Col(
                               dbc.Card([
                                   # dbc.CardHeader(title if title is not None else 'Interval selected:'),
                                   dbc.CardBody(
                                       children=interval_controls_container(
                                           aio_id,
                                           x_axis_type,
                                           x_label=x_label,
                                           extra_dash_components=extra_dash_components,
                                           extra_dash_components2=extra_dash_components2,
                                       ),
                                   ),
                               ]),
                               width=4,
                               align='start',
                           ),
                           dbc.Col(
                               graph(aio_id, figure=figure),
                               width=8,
                               align='start',
                           ),
                       ],
                       align='start',
                   ),
               ],
               fluid=True,
           )
        ]

        kwargs = kwargs.copy()
        kwargs.update(
            children=children,
            fluid=True,
        )
        super().__init__(**kwargs)
