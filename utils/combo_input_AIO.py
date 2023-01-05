from dash import dcc, html, callback, MATCH, ctx, ALL
from dash.dependencies import Input, Output, State


def _component_id(group, aio_id, component_name):
    return {
        'component': 'ComboInputAIO',
        'group': group,
        'aio_id': aio_id,
        'subcomponent': component_name,
    }


def _set_subcomponents_id(components, group, aio_id, input_component_ids):
    if components is None or input_component_ids is None:
        return
    if not isinstance(components, (tuple, list)):
        components = [components]
    if not isinstance(input_component_ids, (tuple, list)):
        input_component_ids = [input_component_ids]

    for component in components:
        component_id = getattr(component, 'id', None)
        if component_id is not None and component_id in input_component_ids:
            component.id = _component_id(group, aio_id, component_id)
        children = getattr(component, 'children', None)
        _set_subcomponents_id(children, group, aio_id, input_component_ids)


def get_combo_input_data_store_id(group):
    return {
        'component': 'ComboInputDataStoreAIO',
        'group': group,
    }


@callback(
    Output(get_combo_input_data_store_id(MATCH), 'data'),
    Input(_component_id(MATCH, ALL, ALL), 'value'),
    State(_component_id(MATCH, ALL, ALL), 'id'),
)
def get_combo_input_values(input_values, input_ids):
    subcomponents = [input_id['subcomponent'] for input_id in input_ids]
    aio_ids = [input_id['aio_id'] for input_id in input_ids]
    combo_input_values = {}
    for subcomponent, aio_id, input_value in zip(subcomponents, aio_ids, input_values):
        combo_input_values.setdefault(aio_id, {})[subcomponent] = input_value
    trigger = ctx.triggered_id
    if trigger is None:
        trigger = {}
    triggered_id = (trigger.get('aio_id'), trigger.get('subcomponent'))
    return {'value': combo_input_values, 'triggered_id': triggered_id}


class ComboInputAIO(html.Div):
    created_groups = set()

    def __init__(
            self,
            children,
            group_id,
            aio_id,
            input_component_ids,
            **kwargs
    ):
        _set_subcomponents_id(children, group_id, aio_id, input_component_ids)

        if group_id not in self.created_groups:
            self.created_groups.add(group_id)
            combo_value = dcc.Store(id=get_combo_input_data_store_id(group_id))
            children = list(children) + [combo_value]

        kwargs = kwargs.copy()
        kwargs.update(children=children)
        super().__init__(**kwargs)
