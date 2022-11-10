import pandas as pd
import xarray as xr
import plotly.express as px
from dash import Dash, html, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from utils.graph_with_horizontal_selection_AIO import GraphWithHorizontalSelectionAIO, figure_data_store_id, variable_label_data_store_id, selected_range_store_id
from utils import charts


ds = xr.load_dataset('/home/wolp/data/tmp/aats-sample-merged-timeseries.nc')
x_min = ds.O3_mean_IAGOS.min().item()
x_max = ds.O3_mean_IAGOS.max().item()

app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
        #'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
)


def get_log_axis_switches(i):
    return dbc.Checklist(
        options=[
            {'label': 'x-axis in log-scale', 'value': 'log_x'},
            {'label': 'y-axis in log-scale', 'value': 'log_y'},
        ],
        value=[],
        id={'type': 'log_scale_switch', 'aio_id': i},
        inline=True,
        switch=True,
        # size='sm',
        # className='mb-3',
    )


# graph_with_horizontal_selection_AIO = GraphWithHorizontalSelectionAIO(
#     'foo',
#     'scalar',
#     x_min=x_min,
#     x_max=x_max,
#     x_label='O3',
#     title='Time interval selected:',
#     extra_dash_components=log_axis_switches
#     #figure=charts.get_avail_data_by_var(ds),
#     #figure=fig,
#     #x_min=min(x).strftime('%Y-%m-%d %H:%M'),
#     #x_max=max(x).strftime('%Y-%m-%d %H:%M'),
# )


@app.callback(
    Output(figure_data_store_id(MATCH), 'data'),
    Input({'type': 'log_scale_switch', 'aio_id': MATCH}, 'value'),
    State(variable_label_data_store_id(MATCH), 'data'),
)
def log_scale_switch_callback(switches, variable_label):
    if switches is None:
        raise PreventUpdate
    log_x = 'log_x' in switches
    log_y = 'log_y' in switches
    x_min = ds[variable_label].min().item()
    x_max = ds[variable_label].max().item()
    new_fig = charts.get_histogram(ds[variable_label], variable_label, log_x=log_x, log_y=log_y)
    figure_data = {
        'fig': new_fig,
        'rng': [x_min, x_max],
    }
    return figure_data


app.layout = html.Div(
    [
        dbc.ListGroup(id='list-group'),
        dbc.Button('Go!', id='go_fig_button_id', n_clicks=0, type='submit')
    ]
)


@app.callback(
    Output('list-group', 'children'),
    Input('go_fig_button_id', 'n_clicks'),
)
def go_callback(go_fig_buttion_n_clicks):
    t_min = pd.Timestamp(ds.time.min().values).strftime('%Y-%m-%d %H:%M')
    t_max = pd.Timestamp(ds.time.max().values).strftime('%Y-%m-%d %H:%M')
    time_filter = GraphWithHorizontalSelectionAIO(
        'time_filter',
        'time',
        x_min=t_min,
        x_max=t_max,
        x_label='time',
        title='Time interval selected:',
        figure=charts.get_avail_data_by_var(ds),
    )

    var_filters = []
    for v, da in ds.data_vars.items():
        var_filter = GraphWithHorizontalSelectionAIO(
            f'{v}_filter',
            'scalar',
            variable_label=v,
            x_min=x_min,
            x_max=x_max,
            x_label=v,
            title=f'{v} interval selected:',
            extra_dash_components=get_log_axis_switches(f'{v}_filter-scalar')
        )
        var_filters.append(var_filter)

    return dbc.ListGroup([time_filter] + var_filters)


if __name__ == "__main__":
    app.run_server(debug=True, host='localhost', port=8055)
