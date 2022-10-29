import sys
from datetime import date
from dash import Dash, dcc, html, callback, MATCH, ALL, ctx
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

ds = xr.load_dataset('/home/wolp/data/tmp/aats-sample-merged-timeseries.nc')

from dash import Dash, html
from jupyter_dash import JupyterDash

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


store_id = _component_id('store')
figure_store_id = _component_id('figure_store')
graph_id = _component_id('graph')
interval_left_end_input_id = _component_id('interval_left_end_input_field')
interval_right_end_input_id = _component_id('interval_right_end_input_field')
interval_controls_listgroup_id = _component_id('interval_controls_listgroup')
remove_interval_button_id = _component_id('remove_interval_button')
new_interval_button_id = _component_id('new_interval_button')
foo_container_id = _component_id('foo_asd')
main_container_id = _component_id('container123')


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
        id=interval_left_end_input_id(aio_id),  # , index),
        type='text',
        debounce=True,
        placeholder='2010-01-01 00:00',
        maxLength=len('2010-01-01 00:00'),
        valid=False,
    )


def interval_right_end_input_field(aio_id): # , index):
    return dbc.Input(
        id=interval_right_end_input_id(aio_id),  # , index),
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
            id=interval_controls_listgroup_id(aio_id),
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
        dbc.Container(
            children='asd',
            id=foo_container_id(aio_id)
        )
    ]


def store(aio_id, data=None):
    return dcc.Store(id=store_id(aio_id), data=data)


def figure_store(aio_id):
    return dcc.Store(id=figure_store_id(aio_id))


class GraphWithHorizontalSelectionAIO(html.Div):
    def __init__(self, aio_id, title=None, figure=Component.UNDEFINED, x_min=None, x_max=None, **kwargs):
        super().__init__(
            children=[
                store(aio_id, [x_min, x_max]),
                figure_store(aio_id),
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
            id=main_container_id(aio_id),
            #className='row', 
            **kwargs
        )


@callback(
    Output(graph_id(MATCH), 'figure'),
    Output(interval_left_end_input_id(MATCH), 'value'),
    Output(interval_left_end_input_id(MATCH), 'valid'),
    Output(interval_right_end_input_id(MATCH), 'value'),
    Output(interval_right_end_input_id(MATCH), 'valid'),
    Output(store_id(MATCH), 'data'),
    Input(figure_store_id(MATCH), 'data'),
    Input(interval_left_end_input_id(MATCH), 'value'),
    Input(interval_right_end_input_id(MATCH), 'value'),
    Input(graph_id(MATCH), 'selectedData'),
    State(graph_id(MATCH), 'figure'),
    State(store_id(MATCH), 'data'),
)
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


@callback(
    Output(foo_container_id(MATCH), 'children'),
    Input(interval_left_end_input_id(MATCH), 'value'),
    Input(interval_right_end_input_id(MATCH), 'value'),
)
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
    x_min=min(x).strftime('%Y-%m-%d %H:%M'),
    x_max=max(x).strftime('%Y-%m-%d %H:%M'),
)


app.layout = html.Div(
    [
        graph_with_horizontal_selection_AIO,
        dbc.Button('Change figure', id='change_fig_button_id', n_clicks=0, type='submit')
    ]
)


@callback(
    Output(figure_store_id(MATCH), 'data'),
    Input('change_fig_button_id', 'n_clicks'),
    prevent_initial_call=True,
)
def change_fig_button_callback(n_clicks):
    global simple_df
    simple_df['x'] += pd.Timedelta('1Y')
    x_min, x_max = min(x), max(x)
    simple_df['y'] += 1
    simple_fig = px.scatter(simple_df, x='x', y='y')
    return {'fig_data': simple_fig['data'], 'x_rng': [x_min.strftime('%Y-%m-%d %H:%M'), x_max.strftime('%Y-%m-%d %H:%M')]}


if __name__ == "__main__":
    app.run_server(debug=True, host='localhost', port=8055)
