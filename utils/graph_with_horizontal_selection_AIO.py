import functools
import dash
from dash import dcc, callback, MATCH, ctx
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
from dash.development.base_component import Component

import utils.dash_dynamic_components as ddc


from log import log_exception

# TODO: for the moment, callback_with_exc_handling is not adapted for the callbacks below;
#  must implement the restrictive pattern-matching with MATCH wildcard, cf:
#   Output 5 (error_message_popup.children@0f7761093e5cf37a5e7e2e5976580b9b)
#   does not have MATCH wildcards on the same keys as
#   Output 0 ({"aio_class":MATCH,"aio_id":MATCH,"component":"GraphWithHorizontalSelectionAIO","subcomponent":"from_input"}.value).
#   MATCH wildcards must be on the same keys for all Outputs.
# from utils.exception_handler import callback_with_exc_handling


_GRAPH_WITH_HORIZONTAL_SELECTION_CONFIG = {
    'autosizable': False,
    'displayModeBar': True,
    # 'fillFrame': True,
    'editSelection': True,
    #'modeBarButtons': [['select2d'], ['toImage']],
    'modeBarButtons': [['toImage']],
    'toImageButtonOptions': {
        'filename': 'atmo-access-plot',
        'format': 'png',
        'height': 800,
    },
    'editable': True,
    'edits': {
        'titleText': True,
        'axisTitleText': True,
        'legendText': True,
        'colorbarTitleText': True,
        'annotationText': False,
        'annotationPosition': False,
    },
    'showAxisDragHandles': False,
    'showAxisRangeEntryBoxes': False,
    'showTips': False,
    'displaylogo': False,
    # 'responsive': True,
}


def _my_explicitize_args(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not Component.UNDEFINED}


def _component_id(component_name):
    def _(aio_id, aio_class):
        component_id = {
            'component': 'GraphWithHorizontalSelectionAIO',
            'subcomponent': component_name,
            'aio_class': aio_class,
            'aio_id': aio_id,
        }
        return component_id

    return _


selected_range_store_id = _component_id('selected_range_store')
figure_data_store_id = _component_id('figure_data_store')
graph_id = _component_id('graph')
from_input_id = _component_id('from_input')
to_input_id = _component_id('to_input')
interval_input_group_id = _component_id('interval_input_group')
reset_selection_button_id = _component_id('reset_selection_button')


@log_exception
#@log_callback_with_ret_value()
def update_graph_figure(fig_data, selected_range):
    if fig_data is None:
        raise dash.exceptions.PreventUpdate

    fig = figure_update_layout(go.Figure(fig_data['fig']))

    if selected_range is not None:
        # variable_label = selected_range['variable_label']
        x0, x1 = selected_range['x_sel_min'], selected_range['x_sel_max']
    else:
        x0, x1 = None, None

    if x0 is not None or x1 is not None:
        if x0 is None:
            x0 = fig_data['rng'][0]
        if x1 is None:
            x1 = fig_data['rng'][1]

        fig.add_shape(
            line={'color': 'grey', 'width': 2, 'dash': 'dot'},
            type='rect',
            xref='x', x0=x0, x1=x1,
            yref='y domain', y0=0, y1=1,
            # fillcolor='blue', opacity=0.1,
        )
    return fig


callback(
    Output(graph_id(MATCH, MATCH), 'figure'),
    Input(figure_data_store_id(MATCH, MATCH), 'data'),
    Input(selected_range_store_id(MATCH, MATCH), 'data'),
)(update_graph_figure)

ddc.dynamic_callback(
    ddc.DynamicOutput(graph_id(MATCH, MATCH), 'figure'),
    ddc.DynamicInput(figure_data_store_id(MATCH, MATCH), 'data'),
    ddc.DynamicInput(selected_range_store_id(MATCH, MATCH), 'data'),
)(update_graph_figure)


@log_exception
# @log_callback_with_ret_value()
def update_from_and_to_input_values(selected_data_on_fig, reset_selection_n_clicks, x0, x1, selected_range, dynamic_component=False):
    if ctx.triggered_id is None:
        raise dash.exceptions.PreventUpdate
        # return None, None, False, False, {'variable_label': selected_range['variable_label'], 'x_sel_min': None, 'x_sel_max': None}

    first_output = ctx.outputs_list[0]
    if dynamic_component:
        if first_output:
            first_output = first_output[0]
        else:
            raise dash.exceptions.PreventUpdate
    x_axis_type = first_output['id']['aio_id'].split('-')[-1]

    if ctx.triggered_id.subcomponent == 'reset_selection_button':
        return None, None, False, False, {'variable_label': selected_range['variable_label'], 'x_sel_min': None, 'x_sel_max': None}
    elif ctx.triggered_id.subcomponent == 'graph':
        # print('selected_data_on_fig =', selected_data_on_fig)
        if selected_data_on_fig is not None and 'range' in selected_data_on_fig:
            new_x0, new_x1 = selected_data_on_fig['range']['x']

            # plotly bug??? selected_data_on_fig['range']['x'] might have new_x0 > new_x1; so need to fix it:
            if new_x0 > new_x1:
                new_x0, new_x1 = new_x1, new_x0

            if x_axis_type == 'time':
                try:
                    new_x0 = np.datetime64(new_x0)
                    new_x1 = np.datetime64(new_x1)
                except ValueError:
                    raise dash.exceptions.PreventUpdate
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


callback(
    Output(from_input_id(MATCH, MATCH), 'value'),
    Output(to_input_id(MATCH, MATCH), 'value'),
    Output(from_input_id(MATCH, MATCH), 'invalid'),
    Output(to_input_id(MATCH, MATCH), 'invalid'),
    Output(selected_range_store_id(MATCH, MATCH), 'data'),
    Input(graph_id(MATCH, MATCH), 'selectedData'),
    Input(reset_selection_button_id(MATCH, MATCH), 'n_clicks'),
    Input(from_input_id(MATCH, MATCH), 'value'),
    Input(to_input_id(MATCH, MATCH), 'value'),
    State(selected_range_store_id(MATCH, MATCH), 'data'),
)(update_from_and_to_input_values)

ddc.dynamic_callback(
    ddc.DynamicOutput(from_input_id(MATCH, MATCH), 'value'),
    ddc.DynamicOutput(to_input_id(MATCH, MATCH), 'value'),
    ddc.DynamicOutput(from_input_id(MATCH, MATCH), 'invalid'),
    ddc.DynamicOutput(to_input_id(MATCH, MATCH), 'invalid'),
    ddc.DynamicOutput(selected_range_store_id(MATCH, MATCH), 'data'),
    ddc.DynamicInput(graph_id(MATCH, MATCH), 'selectedData'),
    ddc.DynamicInput(reset_selection_button_id(MATCH, MATCH), 'n_clicks'),
    ddc.DynamicInput(from_input_id(MATCH, MATCH), 'value'),
    ddc.DynamicInput(to_input_id(MATCH, MATCH), 'value'),
    ddc.DynamicState(selected_range_store_id(MATCH, MATCH), 'data'),
)(functools.partial(update_from_and_to_input_values, dynamic_component=True))



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


def graph(aio_id, aio_class, id_transform, figure=Component.UNDEFINED):
    if figure is not Component.UNDEFINED:
        figure = figure_update_layout(figure)

    return dcc.Graph(
        id=id_transform(graph_id(aio_id, aio_class)),
        config=_GRAPH_WITH_HORIZONTAL_SELECTION_CONFIG,
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


def interval_inputs(aio_id, aio_class, id_transform, x_axis_type, x_label=None):
    if x_axis_type == 'time':
        return dbc.InputGroup(
            [
                dbc.InputGroupText('From'),
                time_input_field(id_transform(from_input_id(aio_id, aio_class))),
                dbc.InputGroupText('to'),
                time_input_field(id_transform(to_input_id(aio_id, aio_class))),
            ],
            id=id_transform(interval_input_group_id(aio_id, aio_class)),
            className='mb-3',
        )
    else:
        return dbc.InputGroup(
            [
                scalar_input_field(id_transform(from_input_id(aio_id, aio_class))),
                dbc.InputGroupText(f'< {x_label if x_label is not None else "x"} <'),
                scalar_input_field(id_transform(to_input_id(aio_id, aio_class))),
            ],
            id=id_transform(interval_input_group_id(aio_id, aio_class)),
            className='mb-3',
        )


def interval_controls_container(
        aio_id,
        aio_class,
        id_transform,
        x_axis_type,
        x_label=None,
        extra_dash_components=None,
        extra_dash_components2=None
):
    if extra_dash_components is None:
        extra_dash_components = []
    if not isinstance(extra_dash_components, (list, tuple)):
        extra_dash_components = [extra_dash_components]
    if extra_dash_components2 is None:
        extra_dash_components2 = []
    if not isinstance(extra_dash_components2, (list, tuple)):
        extra_dash_components2 = [extra_dash_components2]

    return [
        dbc.Row(dbc.Col(interval_inputs(aio_id, aio_class, id_transform, x_axis_type, x_label=x_label))),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        children='Reset filter',
                        id=id_transform(reset_selection_button_id(aio_id, aio_class))
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


def selected_range_store(aio_id, aio_class, id_transform, variable_label=None, x_sel_min=None, x_sel_max=None):
    data = {'variable_label': variable_label, 'x_sel_min': x_sel_min, 'x_sel_max': x_sel_max}
    return dcc.Store(id=id_transform(selected_range_store_id(aio_id, aio_class)), data=data)


def figure_data_store(aio_id, aio_class, id_transform, data=None):
    return dcc.Store(id=id_transform(figure_data_store_id(aio_id, aio_class)), data=data)


class GraphWithHorizontalSelectionAIO:
    def __init__(
            self,
            aio_id,
            x_axis_type,
            aio_class='a-class',
            dynamic_component=False,
            variable_label=None,
            x_min=None,
            x_max=None,
            x_label=None,
            figure=Component.UNDEFINED,
            extra_dash_components=None,
            extra_dash_components2=None,
    ):
        id_transform = ddc.add_active_to_component_id if dynamic_component else lambda _id: _id

        if x_axis_type not in ['time', 'scalar']:
            raise ValueError(f"x_axis_type must be 'time' or 'scalar'; got {x_axis_type}")

        aio_id = aio_id + '-' + x_axis_type
        figure_data = {
            'fig': figure if figure is not Component.UNDEFINED else None,
            'rng': [x_min, x_max],
        }

        _selected_range_store = selected_range_store(
            aio_id,
            aio_class,
            id_transform,
            variable_label=variable_label,
            x_sel_min=None,
            x_sel_max=None,
        )

        _figure_data_store = figure_data_store(
            aio_id,
            aio_class,
            id_transform,
            data=figure_data
        )

        self.data_stores = dbc.Container([_selected_range_store, _figure_data_store])

        self.range_controller = dbc.Card([
            dbc.CardBody(
               children=interval_controls_container(
                   aio_id,
                   aio_class,
                   id_transform,
                   x_axis_type,
                   x_label=x_label,
                   extra_dash_components=extra_dash_components,
                   extra_dash_components2=extra_dash_components2,
               ),
            ),
        ])

        self.graph = graph(aio_id, aio_class, id_transform, figure=figure)

    def get_data_stores(self):
        return self.data_stores

    def get_range_controller(self):
        return self.range_controller

    def get_graph(self):
        return self.graph
