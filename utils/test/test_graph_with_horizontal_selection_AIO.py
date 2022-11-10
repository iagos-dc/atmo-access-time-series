import pandas as pd
import plotly.express as px
from dash import Dash, html, MATCH
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from utils.graph_with_horizontal_selection_AIO import GraphWithHorizontalSelectionAIO, figure_data_store_id


app = Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.FONT_AWESOME
        #'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css',
    ],
)


x = pd.date_range(start='2000', end='2003', tz='UTC', freq='Y')
#x=[1,2,3]
simple_df = pd.DataFrame({'x': x, 'y': [2, 0, 1]})
simple_fig = px.scatter(simple_df, x='x', y='y')


graph_with_horizontal_selection_AIO = GraphWithHorizontalSelectionAIO(
    'foo',
    'time',
    x_label='CO',
    title='Time interval selected:',
    #figure=charts.get_avail_data_by_var(ds),
    figure=simple_fig,
    #x_min=min(x).strftime('%Y-%m-%d %H:%M'),
    #x_max=max(x).strftime('%Y-%m-%d %H:%M'),
)

# @app.callback(
#     Output('output', 'children'),
#     Input('input', 'value'),
# )
# def _(i):
#     return i


app.layout = html.Div(
    [
        graph_with_horizontal_selection_AIO,
        dbc.Button('Change figure', id='change_fig_button_id', n_clicks=0, type='submit')
        #dbc.Input(id='input', type='number', debounce=True),
        #dbc.Container(id='output')
    ]
)


@app.callback(
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
