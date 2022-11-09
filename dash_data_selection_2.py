import sys
from datetime import date
import dash
from dash import Dash, dcc, html, callback, MATCH, ALL, ctx
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import xarray as xr
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import json

from dash.development.base_component import Component

sys.path.append('/home/wolp/PycharmProjects/atmo-access-time-series')

import data_processing
from utils import charts
import logging
from log import log_args, start_logging, start_logging_callbacks, log_callback, log_callback_with_ret_value


ds = xr.load_dataset('/home/wolp/data/tmp/aats-sample-merged-timeseries.nc')

from dash import Dash, html
from jupyter_dash import JupyterDash

start_logging_callbacks('/home/wolp/PycharmProjects/atmo-access-time-series/log/log.mmap')

app = Dash(
    __name__, 
    external_stylesheets=[
        dbc.themes.BOOTSTRAP, 
        dbc.icons.FONT_AWESOME
        #'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
)


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
    #print(f'update_graph_figure(\n  fig_data={fig_data}\n  selected_range={selected_range}\n  fig={fig}\n)')
    if fig is None:
        fig = go.Figure()

    if fig_data is not None:
        fig['data'] = fig_data

    fig['layout'].pop('shapes', None)
    fig['layout'].pop('selections', None)
    x0, x1 = selected_range if selected_range is not None else (None, None)
    if x0 is not None or x1 is not None:
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
    Output(from_input_id(MATCH), 'valid'),
    Output(to_input_id(MATCH), 'valid'),
    Input(graph_id(MATCH), 'selectedData'),
    Input(reset_selection_button_id(MATCH), 'n_clicks'),
    Input(from_input_id(MATCH), 'value'),
    Input(to_input_id(MATCH), 'value'),
)
@log_callback_with_ret_value()
def update_from_and_to_input_values(selected_data_on_fig, reset_selection_n_clicks, x0, x1):
    if ctx.triggered_id is None or ctx.triggered_id.subcomponent == 'reset_selection_button':
        return None, None, None, None
    elif ctx.triggered_id.subcomponent == 'graph':
        if selected_data_on_fig is not None and 'range' in selected_data_on_fig:
            new_x0, new_x1 = selected_data_on_fig['range']['x']
            new_x0 = pd.Timestamp(new_x0).strftime('%Y-%m-%d %H:%M')
            new_x1 = pd.Timestamp(new_x1).strftime('%Y-%m-%d %H:%M')
            return new_x0, new_x1, True, True
        else:
            raise PreventUpdate
    elif ctx.triggered_id.subcomponent in ['from_input', 'to_input']:
        x0_valid = _valid(x0)
        x1_valid = _valid(x1)
        x0 = pd.Timestamp(x0).strftime('%Y-%m-%d %H:%M') if x0_valid else dash.no_update
        x1 = pd.Timestamp(x1).strftime('%Y-%m-%d %H:%M') if x1_valid else dash.no_update
        return x0, x1, x0_valid, x1_valid


def _valid(x):
    try:
        x = pd.Timestamp(x)
    except ValueError:
        return False
    if pd.isnull(x):
        return None
    return True


# @callback(
#     Output(from_input_id(MATCH), 'valid'),
#     Input(from_input_id(MATCH), 'value'),
# )
# @log_callback_with_ret_value()
# def validate_from_input(x):
#     return _valid(x)
#
#
# @callback(
#     Output(to_input_id(MATCH), 'valid'),
#     Input(to_input_id(MATCH), 'value'),
# )
# @log_callback_with_ret_value()
# def validate_to_input(x):
#     return _valid(x)


@callback(
    Output(selected_range_store_id(MATCH), 'data'),
    Input(from_input_id(MATCH), 'value'),
    Input(to_input_id(MATCH), 'value'),
    Input(from_input_id(MATCH), 'valid'),
    Input(to_input_id(MATCH), 'valid'),
    State(selected_range_store_id(MATCH), 'data'),
)
@log_callback_with_ret_value()
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


# @callback(
#     Output(graph_id(MATCH), 'figure'),
#     Output(selected_range_store_id(MATCH), 'data'),
#     Output(from_input_id(MATCH), 'value'),
#     Output(to_input_id(MATCH), 'value'),
#     Output(to_input_id(MATCH), 'valid'),
#     Output(from_input_id(MATCH), 'valid'),
#     Input(figure_data_store_id(MATCH), 'data'),
#     Input(graph_id(MATCH), 'selectedData'),
#     Input(reset_selection_button_id(MATCH), 'n_clicks'),
#     Input(from_input_id(MATCH), 'value'),
#     Input(to_input_id(MATCH), 'value'),
#     State(selected_range_store_id(MATCH), 'data'),
#     State(graph_id(MATCH), 'figure'),
#     State(from_input_id(MATCH), 'valid'),
#     State(to_input_id(MATCH), 'valid'),
# )
# @log_callback_with_ret_value()
# def combo_callback(fig_data, selected_data, reset_selection_n_clicks, x0, x1, selected_range, fig, x0_valid, x1_valid):
#     if ctx.triggered_id is None:
#         return None, [None, None], None, None, None, None
#     elif ctx.triggered_id.subcomponent == 'figure_data_store':
#         pass
#     elif ctx.triggered_id.subcomponent == 'graph':
#         pass
#     elif ctx.triggered_id.subcomponent == 'reset_selection_button':
#         pass
#     elif ctx.triggered_id.subcomponent == 'from_input':
#         pass
#     elif ctx.triggered_id.subcomponent == 'to_input':
#         pass
#     else:
#         raise RuntimeError(f'unknown callback trigger: ctx.triggered_id.subcomponent={ctx.triggered_id.subcomponent}, ctx.triggered_id={ctx.triggered_id}, ctx={ctx}')


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


def interval_left_end_input_field(aio_id): # , index):
    return dbc.Input(
        id=from_input_id(aio_id),  # , index),
        type='text',
        debounce=True,
        placeholder='2010-01-01 00:00',
        maxLength=len('2010-01-01 00:00'),
        valid=False,
    )


def interval_right_end_input_field(aio_id): # , index):
    return dbc.Input(
        id=to_input_id(aio_id),  # , index),
        type='text',
        debounce=True,
        placeholder='2010-11-30 23:59',
        maxLength=len('2010-11-30 23:59'),
        valid=False,
    )


def interval_inputs(aio_id): # , index):
    return dbc.InputGroup(
        [
            dbc.InputGroupText('From'),
            interval_left_end_input_field(aio_id),
            dbc.InputGroupText('to'),
            interval_right_end_input_field(aio_id),
        ],
        className='mb-3',
    )


def interval_controls_container(aio_id):
    return [
        dbc.ListGroup(
            [
                interval_inputs(aio_id), #, 0),
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
        dbc.Container(
            children='asd',
            id=foo_container_id(aio_id)
        )
    ]


def selected_range_store(aio_id, data=None):
    return dcc.Store(id=selected_range_store_id(aio_id), data=data)


def figure_data_store(aio_id, data=None):
    return dcc.Store(id=figure_data_store_id(aio_id), data=data)


class GraphWithHorizontalSelectionAIO(html.Div):
    def __init__(self, aio_id, title=None, figure=Component.UNDEFINED, **kwargs):
        super().__init__(
            children=[
                selected_range_store(aio_id, [None, None]),
                figure_data_store(aio_id, figure['data'] if figure is not Component.UNDEFINED else None),
                dbc.Container(
                    [
                        html.H1('GraphWithHorizontalSelectionAIO'),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card([
                                        dbc.CardHeader(title if title is not None else 'Interval selected:'),
                                        dbc.CardBody(
                                            children=interval_controls_container(aio_id),
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


# @callback(
#     Output(graph_id(MATCH), 'figure'),
#     Output(interval_left_end_input_id(MATCH), 'value'),
#     Output(interval_left_end_input_id(MATCH), 'valid'),
#     Output(interval_right_end_input_id(MATCH), 'value'),
#     Output(interval_right_end_input_id(MATCH), 'valid'),
#     Output(store_id(MATCH), 'data'),
#     Input(figure_store_id(MATCH), 'data'),
#     Input(interval_left_end_input_id(MATCH), 'value'),
#     Input(interval_right_end_input_id(MATCH), 'value'),
#     Input(graph_id(MATCH), 'selectedData'),
#     State(graph_id(MATCH), 'figure'),
#     State(store_id(MATCH), 'data'),
# )
def _xaxis_range_selection(fig_data, x0, x1, selected_data, fig, store_data):
    print(f'ctx={ctx.triggered_id.subcomponent if ctx.triggered_id else None},{ctx.triggered_prop_ids}, x0={x0}, x1={x1}, \nlen(selectedData)={len(selected_data.get("points", [])) if selected_data else 0}')

    # if ctx.triggered_id is None or ctx.triggered_id.subcomponent == 'figure_store':
    #     fig = figure_update_layout(fig)
    #     fig = apply_selection(fig, x0, x1)
    #     return fig, x0

    if ctx.triggered_id is not None and ctx.triggered_id.subcomponent == 'graph':
        figure_layout_selections = fig['layout'].get('selections', [])
        figure_layout_selections = [sel for sel in figure_layout_selections if sel.get('type') == 'rect']
        if figure_layout_selections:
            last_selection = figure_layout_selections[-1]
            #print(last_selection)
            fig['layout']['selections'] = [last_selection]
            x0, x1 = pd.Timestamp(last_selection['x0']), pd.Timestamp(last_selection['x1'])
            x0, x1 = min(x0, x1), max(x0, x1)
            return fig, x0.strftime('%Y-%m-%d %H:%M'), True, x1.strftime('%Y-%m-%d %H:%M'), True, store_data,
        else:
            return fig, None, False, None, False, store_data,
    elif ctx.triggered_id is None or ctx.triggered_id.subcomponent in ['interval_left_end_input_field', 'interval_right_end_input_field']:
        x0_valid, x1_valid, fig = validate_interval_input(x0, x1, fig, *store_data)
        return fig, x0, x0_valid, x1, x1_valid, store_data,
    elif ctx.triggered_id.subcomponent == 'figure_store':
        x_min, x_max = fig_data['x_rng']
        fig['data'] = fig_data['fig_data']
        return fig, x0, True, x1, True, [x_min, x_max]
    else:
        raise RuntimeError(f'unknown context triggered id: {ctx.triggered_id}')


def validate_interval_input(x0, x1, fig, x_min, x_max):
    def valid(x):
        try:
            x = pd.Timestamp(x)
        except ValueError:
            return False
        if pd.isnull(x):
            return None
        return True

    are_valid = [valid(x0), valid(x1)]
    if all(v is not False for v in are_valid):
        selection = {
            'xref': 'x', 'yref': 'y', 'line': {'width': 1, 'dash': 'dot'}, 'type': 'rect',
            'x0': x0 if are_valid[0] else x_min, 'y0': 2.141804788213628,
            'x1': x1 if are_valid[1] else x_max, 'y1': -0.14180478821362844,
        }
        fig['layout']['selections'] = [selection]

    return list(map(bool, are_valid)) + [fig]

#@callback(
#    Output(ids.interval_left_end_input_field(MATCH), 'value'),
#    Input(ids.interval_left_end_input_field(MATCH), 'valid'),
#    State(ids.interval_left_end_input_field(MATCH), 'value'),
#)
#def update_value_of_interval_left_end_input_field(ok, v):
#    return v.strftime('%Y-%m-%d %H:%M:%S')

# @callback(
#     Output(interval_right_end_input_id(MATCH), 'valid'),
#     #Output(ids.interval_right_end_input_field(MATCH), 'value'),
#     Input(interval_right_end_input_id(MATCH), 'value'),
# )
# def validate_interval_right_end_input_field(v):
#     try:
#         v2 = pd.Timestamp(v)
#     except ValueError:
#         return False
#     if pd.isnull(v2):
#         return False
#     return True


# @callback(
#     Output(foo_container_id(MATCH), 'children'),
#     Input(interval_left_end_input_id(MATCH), 'value'),
#     Input(interval_right_end_input_id(MATCH), 'value'),
# )
def foo_callback(left_end, right_end):
    return f'left={left_end}, right={right_end}'


x = pd.date_range(start='2000', end='2003', tz='UTC', freq='Y')
#x=[1,2,3]
simple_df = pd.DataFrame({'x': x, 'y': [2, 0, 1]})
simple_fig = px.scatter(simple_df, x='x', y='y')

graph_with_horizontal_selection_AIO = GraphWithHorizontalSelectionAIO(
    'foo',
    title='Time interval selected:',
    #figure=charts.get_avail_data_by_var(ds),
    figure=simple_fig,
    #x_min=min(x).strftime('%Y-%m-%d %H:%M'),
    #x_max=max(x).strftime('%Y-%m-%d %H:%M'),
)


app.layout = html.Div(
    [
        graph_with_horizontal_selection_AIO,
        dbc.Button('Change figure', id='change_fig_button_id', n_clicks=0, type='submit')
    ]
)


@callback(
    Output(figure_data_store_id(MATCH), 'data'),
    Input('change_fig_button_id', 'n_clicks'),
    prevent_initial_call=True,
)
def change_fig_button_callback(n_clicks):
    global simple_df
    simple_df['x'] += pd.Timedelta('1Y')
    x_min, x_max = min(x), max(x)
    simple_df['y'] += 1
    simple_fig = px.scatter(simple_df, x='x', y='y')
    return simple_fig['data']
    #return {'fig_data': simple_fig['data'], 'x_rng': [x_min.strftime('%Y-%m-%d %H:%M'), x_max.strftime('%Y-%m-%d %H:%M')]}


if __name__ == "__main__":
    #start_logging('/home/wolp/PycharmProjects/atmo-access-time-series/log/log2.txt', logging_level=logging.INFO)
    #start_logging(logging_level=logging.INFO)
    app.run_server(debug=True, host='localhost', port=8055)
