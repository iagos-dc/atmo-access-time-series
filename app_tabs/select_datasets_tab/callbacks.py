import json
import toolz
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import werkzeug.utils
from dash import callback, Output, Input, State, dcc

import data_access
import data_processing
import data_processing.utils
from app_tabs.common.data import station_by_shortnameRI
from app_tabs.common.layout import DATASETS_STORE_ID, INTEGRATE_DATASETS_REQUEST_ID
from app_tabs.select_datasets_tab.layout import GANTT_GRAPH_ID, GANTT_VIEW_RADIO_ID, DATASETS_TABLE_ID, \
    DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID, QUICKLOOK_POPUP_ID, SELECT_DATASETS_BUTTON_ID
from log import logger, log_exception
from utils import charts


@callback(
    Output(GANTT_GRAPH_ID, 'figure'),
    Output(GANTT_GRAPH_ID, 'selectedData'),
    Input(GANTT_VIEW_RADIO_ID, 'value'),
    Input(DATASETS_STORE_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
def get_gantt_figure(gantt_view_type, datasets_json):
    selectedData = {'points': []}

    if datasets_json is None:
       return {}, selectedData   # empty figure; TODO: is it a right way?

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    datasets_df = datasets_df.join(station_by_shortnameRI['station_fullname'], on='platform_id_RI')  # column 'station_fullname' joined to datasets_df

    if len(datasets_df) == 0:
       return {}, selectedData   # empty figure; TODO: is it a right way?

    if gantt_view_type == 'compact':
        fig = charts._get_timeline_by_station(datasets_df)
    else:
        fig = charts._get_timeline_by_station_and_vars(datasets_df)
    fig.update_traces(
        selectedpoints=[],
        #mode='markers+text', marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 20},
        unselected={'marker': {'opacity': 0.4}, }
    )
    return fig, selectedData


@callback(
    Output(DATASETS_TABLE_ID, 'columns'),
    Output(DATASETS_TABLE_ID, 'data'),
    Output(DATASETS_TABLE_ID, 'selected_rows'),
    Output(DATASETS_TABLE_ID, 'selected_row_ids'),
    Input(GANTT_GRAPH_ID, 'selectedData'),
    Input(DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    State(DATASETS_STORE_ID, 'data'),
    State(DATASETS_TABLE_ID, 'selected_row_ids'),
    prevent_initial_call=True,
)
@log_exception
def datasets_as_table(gantt_figure_selectedData, datasets_table_checklist_all_none_switch,
                      datasets_json, previously_selected_row_ids):
    table_col_ids = ['eye', 'title', 'var_codes_filtered', 'RI', 'long_name', 'platform_id', 'time_period_start', 'time_period_end',
                     #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                     ]
    table_col_names = ['', 'Title', 'Variables', 'RI', 'Station', 'Station code', 'Start', 'End',
                       #_#'url', 'ecv_variables', 'ecv_variables_filtered', 'std_ecv_variables_filtered', 'var_codes', 'platform_id_RI'
                       ]
    table_columns = [{'name': name, 'id': i} for name, i in zip(table_col_names, table_col_ids)]
    # on rendering HTML snipplets in DataTable cells:
    # https://github.com/plotly/dash-table/pull/916
    table_columns[0]['presentation'] = 'markdown'

    if datasets_json is None:
        return table_columns, [], [], []

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    if len(datasets_df) > 0:
        datasets_df['time_period_start'] = datasets_df['time_period_start'].dt.strftime('%Y-%m-%d')
        datasets_df['time_period_end'] = datasets_df['time_period_end'].dt.strftime('%Y-%m-%d')

    datasets_df = datasets_df.join(station_by_shortnameRI['long_name'], on='platform_id_RI')

    # filter on selected timeline bars on the Gantt figure
    if gantt_figure_selectedData and 'points' in gantt_figure_selectedData:
        datasets_indices = []
        for timeline_bar in gantt_figure_selectedData['points']:
            datasets_indices.extend(timeline_bar['customdata'][0])
        datasets_df = datasets_df.iloc[datasets_indices]

    # on rendering HTML snipplets in DataTable cells:
    # https://github.com/plotly/dash-table/pull/916
    datasets_df['eye'] = '<i class="fa fa-eye"></i>'

    table_data = datasets_df[['id'] + table_col_ids].to_dict(orient='records')

    # see here for explanation how dash.callback_context works
    # https://community.plotly.com/t/select-all-rows-in-dash-datatable/41466/2
    # TODO: this part needs to be checked and polished;
    # TODO: e.g. is the manual synchronization between selected_rows and selected_row_ids needed?
    trigger = dash.callback_context.triggered[0]['prop_id'].split('.')[0]
    if trigger == DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID:
        if datasets_table_checklist_all_none_switch:
            selected_rows = list(range(len(table_data)))
        else:
            selected_rows = []
        selected_row_ids = datasets_df['id'].iloc[selected_rows].to_list()
    else:
        if previously_selected_row_ids is None:
            previously_selected_row_ids = []
        selected_row_ids = sorted(set(previously_selected_row_ids) & set(datasets_df['id'].to_list()))
        idx = pd.DataFrame({'idx': datasets_df['id'], 'n': range(len(datasets_df['id']))}).set_index('idx')
        idx = idx.loc[selected_row_ids]
        selected_row_ids = idx.index.to_list()
        selected_rows = idx['n'].to_list()
    return table_columns, table_data, selected_rows, selected_row_ids


@callback(
    Output(QUICKLOOK_POPUP_ID, 'children'),
    Input(DATASETS_TABLE_ID, 'active_cell'),
    State(DATASETS_STORE_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
def popup_graphs(active_cell, datasets_json):
    if datasets_json is None or active_cell is None:
        return None

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    ds_md = datasets_df.loc[active_cell['row_id']]

    if 'selector' in ds_md and isinstance(ds_md['selector'], str) and len(ds_md['selector']) > 0:
        selector = ds_md['selector']
    else:
        selector = None

    print('selector=', selector)
    try:
        ri = ds_md['RI']
        url = ds_md['url']
        selector = ds_md['selector'] if 'selector' in ds_md else None
        req = data_processing.ReadDataRequest(ri, url, ds_md, selector=selector)
        da_by_var = req.compute()
        # da_by_var = data_access.read_dataset(ds_md['RI'], ds_md['url'], ds_md)
        # for v, da in da_by_var.items():
        #     da.to_netcdf(CACHE_DIR / 'tmp' / f'{v}.nc')
        ds_exc = None
    except Exception as e:
        da_by_var = None
        ds_exc = e

    if da_by_var is not None:
        da_by_var = toolz.valfilter(lambda da: da.squeeze().ndim == 1, da_by_var)
        if len(da_by_var) > 0:
            series_by_var = toolz.valmap(
                # due to peformance, make a random subsampling of the timeseries
                lambda da: data_processing.utils.subsampling(da.to_series(), n=3000),
                da_by_var
            )
            fig = charts.multi_line(series_by_var, width=1000)
            fig = charts.add_watermark(fig)
            fig.update_layout(
                legend=dict(orientation='h'),
                title=ds_md['title'],
                hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
            )
            ds_plot = dcc.Graph(
                id='quick-plot',
                figure=fig,
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'scrollZoom': False,
                }
            )
        else:
            ds_plot = None
    else:
        ds_plot = repr(ds_exc)

    popup = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle(ds_md['title'])),
            dbc.ModalBody(children=[
                ds_plot,
                # html.Button('Download CSV', id='btn_csv'),
                # dcc.Download(id='download_csv'),
            ]),
        ],
        id="modal-xl",
        size="xl",
        is_open=True,
    )

    return popup #, ds_md.to_json(orient='index', date_format='iso')


# @callback(
#     Output('download_csv', 'data'),
#     Input('btn_csv', 'n_clicks'),
#     State(DATASET_MD_STORE_ID, 'data'),
#     prevent_initial_call=True,
# )
@log_exception
def download_csv(n_clicks, ds_md_json):
    try:
        s = pd.Series(json.loads(ds_md_json))
        ds = data_access.read_dataset(s['RI'], s['url'], s)
        df = ds.reset_coords(drop=True).to_dataframe()
        download_filename = werkzeug.utils.secure_filename(s['title'] + '.csv')
        return dcc.send_data_frame(df.to_csv, download_filename)
    except Exception as e:
        logger().exception(f'Failed to download the dataset {ds_md_json}', exc_info=e)


@callback(
    Output(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    Input(SELECT_DATASETS_BUTTON_ID, 'n_clicks'),
    State(DATASETS_STORE_ID, 'data'),
    State(DATASETS_TABLE_ID, 'selected_row_ids'),
    prevent_initial_call=True,
)
@log_exception
def select_datasets(n_clicks, datasets_json, selected_row_ids):
    if datasets_json is None or selected_row_ids is None:
        return None

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    ds_md = datasets_df.loc[selected_row_ids]
    read_dataset_requests = []
    for idx, ds_metadata in ds_md.iterrows():
        ri = ds_metadata['RI']
        url = ds_metadata['url']
        selector = ds_metadata['selector'] if 'selector' in ds_metadata else None
        req = data_processing.ReadDataRequest(ri, url, ds_metadata, selector=selector)
        # req.compute()  ###
        read_dataset_requests.append(req)

    req = data_processing.IntegrateDatasetsRequest(read_dataset_requests)
    # TODO: do it asynchronously? will it work with dash/flask? look at options of @app.callback decorator (background=True, ???)
    req.compute()
    return req.to_dict()
