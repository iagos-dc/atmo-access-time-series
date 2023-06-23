import functools
from dash import html, Output, no_update
from dash.exceptions import DashException
import dash_bootstrap_components as dbc
from utils import dash_dynamic_components as ddc


ERROR_MESSAGE_POPUP_ID = 'error_message_popup'
ERROR_MESSAGE_POPUP_ID2 = 'error_message_popup2'


class AppException(Exception):
    pass


def get_error_message(msg):
    popup = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(msg)),
        ],
        id="modal-xl",
        size="xl",
        is_open=True,
    )
    return popup


def handle_exception(callback_decorator, *default_outputs):
    def callback_decorator_with_exception_handler(*args, **kwargs):
        no_outputs = len([arg for arg in args if isinstance(arg, Output)])
        no_default_outputs = len(default_outputs)
        if no_default_outputs > no_outputs:
            raise ValueError(f'len(default_outputs)={no_default_outputs} must be <= no_outputs={no_outputs}')

        new_args = (Output(ERROR_MESSAGE_POPUP_ID, 'children', allow_duplicate=True), ) + args
        new_kwargs = dict(kwargs)
        new_kwargs.update(prevent_initial_call=True)

        def callback_func_transform(callback_func):
            @functools.wraps(callback_func)
            def callback_func_with_exception_handling(*callback_args):
                error_message = None
                try:
                    callback_result = callback_func(*callback_args)
                except DashException:
                    raise
                except Exception as e:
                    error_message = get_error_message(e.args)
                if error_message is None:
                    return (no_update, ) + callback_result
                else:
                    outputs = default_outputs + (no_update, ) * (no_outputs - no_default_outputs)
                    return (error_message, ) + outputs
            return callback_decorator(*new_args, **new_kwargs)(callback_func_with_exception_handling)

        return callback_func_transform
    return callback_decorator_with_exception_handler


error_message_popup = html.Div(id=ERROR_MESSAGE_POPUP_ID)
error_message_popup2 = html.Div(id=ddc.add_active_to_component_id(ERROR_MESSAGE_POPUP_ID2))
