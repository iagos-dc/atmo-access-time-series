import pandas as pd
import xarray as xr
import plotly.express as px
from dash import Dash, html, MATCH
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from utils.graph_with_horizontal_selection_AIO import GraphWithHorizontalSelectionAIO, figure_data_store_id
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


#fig = charts.get_avail_data_by_var(ds)
fig = charts.get_histogram(ds.O3_mean_IAGOS, 'O3')


log_axis_switches = dbc.Checklist(
    options=[
        {'label': 'x-axis in log-scale', 'value': 'log_x'},
        {'label': 'y-axis in log-scale', 'value': 'log_y'},
    ],
    value=[],
    id='log_scale_switch',
    inline=True,
    switch=True,
    # size='sm',
    # className='mb-3',
)


graph_with_horizontal_selection_AIO = GraphWithHorizontalSelectionAIO(
    'foo',
    'scalar',
    x_min=x_min,
    x_max=x_max,
    x_label='O3',
    title='Time interval selected:',
    extra_dash_components=log_axis_switches
    #figure=charts.get_avail_data_by_var(ds),
    #figure=fig,
    #x_min=min(x).strftime('%Y-%m-%d %H:%M'),
    #x_max=max(x).strftime('%Y-%m-%d %H:%M'),
)


@app.callback(
    Output(figure_data_store_id('foo-scalar'), 'data'),
    Input('log_scale_switch', 'value')
)
def log_scale_switch_callback(switches):
    if switches is None:
        raise PreventUpdate
    log_x = 'log_x' in switches
    log_y = 'log_y' in switches
    new_fig = charts.get_histogram(ds.O3_mean_IAGOS, 'O3', log_x=log_x, log_y=log_y)
    figure_data = {
        'fig': new_fig,
        'rng': [x_min, x_max],
    }
    return figure_data


app.layout = html.Div(
    [
        dbc.Card(
            children=[
                graph_with_horizontal_selection_AIO,
            ],
            body=True,
        )
        #dbc.Button('Change figure', id='change_fig_button_id', n_clicks=0, type='submit')
        #dbc.Input(id='input', type='number', debounce=True),
        #dbc.Container(id='output')
    ]
)


if __name__ == "__main__":
    #start_logging('/home/wolp/PycharmProjects/atmo-access-time-series/log/log2.txt', logging_level=logging.INFO)
    #start_logging(logging_level=logging.INFO)
    app.run_server(debug=True, host='localhost', port=8055)
