import json
import toolz
import numpy as np
import pandas as pd
import dash
import dash_bootstrap_components as dbc
import werkzeug.utils
from dash import callback, Output, Input, State, Patch, dcc

import data_access
import data_processing
import data_processing.utils
from app_tabs.common.data import station_by_shortnameRI
from app_tabs.common.layout import APP_TABS_ID, DATASETS_STORE_ID, INTEGRATE_DATASETS_REQUEST_ID, \
    GANTT_SELECTED_DATASETS_IDX_STORE_ID, GANTT_SELECTED_BARS_STORE_ID, FILTER_DATA_TAB_VALUE, SELECT_DATASETS_TAB_VALUE
from app_tabs.select_datasets_tab.layout import GANTT_GRAPH_ID, GANTT_VIEW_RADIO_ID, DATASETS_TABLE_ID, \
    DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID, QUICKLOOK_POPUP_ID, SELECT_DATASETS_BUTTON_ID, \
    RESET_DATASETS_SELECTION_BUTTON_ID, SELECTED_GANTT_OPACITY, UNSELECTED_GANTT_OPACITY, BAR_UNSELECTED, \
    BAR_PARTIALLY_SELECTED, BAR_SELECTED, OPACITY_BY_BAR_SELECTION_STATUS
from log import logger, log_exception, log_exectime
from utils import charts
from utils.exception_handler import handle_exception


@callback(
    Output(SELECT_DATASETS_TAB_VALUE, 'disabled'),
    Input(DATASETS_STORE_ID, 'data'),
)
@log_exception
def enable_select_datasets_tab(datasets_json):
    return datasets_json is None


# @callback(
#     Output(GANTT_SELECTED_ITEMS_STORE_ID, 'data'),
#     Input(GANTT_GRAPH_ID, 'selectedData'),
#     Input(DATASETS_STORE_ID, 'data'),
# )
@log_exception
def get_gantt_selected_items_store(gantt_figure_selectedData, datasets_json):
    dash_ctx = list(dash.ctx.triggered_prop_ids.values())
    print(f'get_gantt_selected_items_store::dash_ctx={dash_ctx}')
    if DATASETS_STORE_ID in dash_ctx and GANTT_GRAPH_ID not in dash_ctx:
        return {'points': []}
    else:
        return gantt_figure_selectedData


_callback_with_exc_handling = handle_exception(callback)
_custom_callback_with_exc_handling = handle_exception(callback, dash.no_update, dash.no_update, dash.no_update, None)

@_custom_callback_with_exc_handling(
    Output(GANTT_SELECTED_DATASETS_IDX_STORE_ID, 'data'),
    Output(GANTT_SELECTED_BARS_STORE_ID, 'data'),
    Output(GANTT_GRAPH_ID, 'figure', allow_duplicate=True),
    Output(GANTT_GRAPH_ID, 'clickData'),
    Input(RESET_DATASETS_SELECTION_BUTTON_ID, 'n_clicks'),
    Input(GANTT_GRAPH_ID, 'clickData'),
    State(GANTT_SELECTED_DATASETS_IDX_STORE_ID, 'data'),
    State(GANTT_SELECTED_BARS_STORE_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
def update_selection_on_gantt_graph(
        reset_datasets_selection_button,
        click_data,
        selected_datasets,
        selected_gantt_bars,
):
    dash_ctx = list(dash.ctx.triggered_prop_ids.values())

    print(f'click_data={click_data}')
    print(f'selected_gantt_bars={selected_gantt_bars}')

    if selected_gantt_bars is None:
        # it means that the figure is not initialized (= it is None)
        raise dash.exceptions.PreventUpdate

    patched_fig = Patch()

    if RESET_DATASETS_SELECTION_BUTTON_ID in dash_ctx:
        selected_gantt_bars_new = []
        for trace_no, selected_gantt_bars_in_category in enumerate(selected_gantt_bars):
            selected_gantt_bars_in_category = np.asarray(selected_gantt_bars_in_category)
            selected_gantt_bars_in_category[:] = BAR_UNSELECTED
            selected_gantt_bars_new.append(selected_gantt_bars_in_category)
            patched_fig['data'][trace_no]['marker']['opacity'] = pd.Series(selected_gantt_bars_in_category).map(OPACITY_BY_BAR_SELECTION_STATUS).values
            # patched_fig['data'][trace_no]['marker']['opacity'] = np.where(
            #     selected_gantt_bars_in_category,
            #     SELECTED_GANTT_OPACITY,
            #     UNSELECTED_GANTT_OPACITY
            # )

        return [], selected_gantt_bars_new, patched_fig, None

    if GANTT_GRAPH_ID in dash_ctx and click_data is not None:
        print(f'click_data={click_data}')
        try:
            click_datasets = click_data['points'][0]['customdata'][0]
            trace_no = click_data['points'][0]['curveNumber']
            point_index = click_data['points'][0]['pointIndex']
        except Exception:
            raise dash.exceptions.PreventUpdate
        if selected_datasets is None:
            selected_datasets = []

        if selected_gantt_bars[trace_no][point_index] != BAR_SELECTED:
            selected_gantt_bars[trace_no][point_index] = BAR_SELECTED
            click_datasets_set = set(click_datasets)
            selected_datasets2 = [ds_idx for ds_idx in selected_datasets if ds_idx not in click_datasets_set]
            selected_datasets = click_datasets + selected_datasets2
            patched_fig['data'][trace_no]['marker']['opacity'][point_index] = SELECTED_GANTT_OPACITY
        else:
            selected_gantt_bars[trace_no][point_index] = BAR_UNSELECTED
            click_datasets_set = set(click_datasets)
            selected_datasets = [ds_idx for ds_idx in selected_datasets if ds_idx not in click_datasets_set]
            patched_fig['data'][trace_no]['marker']['opacity'][point_index] = UNSELECTED_GANTT_OPACITY

        return selected_datasets, selected_gantt_bars, patched_fig, None

    raise dash.exceptions.PreventUpdate


@callback(
    Output(GANTT_GRAPH_ID, 'figure'),
    Output(GANTT_SELECTED_DATASETS_IDX_STORE_ID, 'data', allow_duplicate=True),
    Output(GANTT_SELECTED_BARS_STORE_ID, 'data', allow_duplicate=True),
    Input(GANTT_VIEW_RADIO_ID, 'value'),
    Input(DATASETS_STORE_ID, 'data'),
    Input(APP_TABS_ID, 'value'),  # dummy trigger; it is a way to workaround plotly bug of badly resized figures
    State(GANTT_SELECTED_DATASETS_IDX_STORE_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
#@log_exectime
def get_gantt_figure(gantt_view_type, datasets_json, app_tab_value, selected_datasets):
    # TODO: transfer gantt_figure_selectedData to that of output (now it is cleared)
    if app_tab_value != SELECT_DATASETS_TAB_VALUE:
        raise dash.exceptions.PreventUpdate

    # reset selected datasets after "Search datasets" button clicked
    ctx = list(dash.ctx.triggered_prop_ids.values())
    if DATASETS_STORE_ID in ctx:
        selected_datasets = None
        selected_datasets_output = None
    else:
        selected_datasets_output = dash.no_update

    if datasets_json is None:
        return charts.empty_figure(height=400), selected_datasets_output, None

    selected_datasets = set(selected_datasets) if selected_datasets is not None else set()

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    datasets_df = datasets_df.join(station_by_shortnameRI['station_fullname'], on='platform_id_RI')  # column 'station_fullname' joined to datasets_df

    if len(datasets_df) == 0:
       return charts.empty_figure(height=400), selected_datasets_output, None

    if gantt_view_type == 'compact':
        fig = charts._get_timeline_by_station(datasets_df)
    else:
        fig = charts._get_timeline_by_station_and_vars(datasets_df)

    fig.update_traces(
        selected={'marker_opacity': SELECTED_GANTT_OPACITY},
        unselected={'marker_opacity': UNSELECTED_GANTT_OPACITY},
        # marker={'opacity': UNSELECTED_GANTT_OPACITY},
        # unselected={'marker': {'opacity': 0.4}, }
        # mode='markers+text', marker={'color': 'rgba(0, 116, 217, 0.7)', 'size': 20},
    )

    # TODO: do it properly, using update_traces, etc.
    fig_data = fig['data']
    gantt_selected_bars = []

    def get_bar_selection_status(datasets_idx):
        datasets_idx = set(datasets_idx)
        if datasets_idx.isdisjoint(selected_datasets):
            return BAR_UNSELECTED
        elif datasets_idx <= selected_datasets:
            return BAR_SELECTED
        else:
            return BAR_PARTIALLY_SELECTED

    for fig_data_category in fig_data:
        fig_data_category_customdata = fig_data_category['customdata']
        fig_data_category_len = len(fig_data_category_customdata)
        gantt_selected_bars_for_category = [
            get_bar_selection_status(fig_data_category_customdata_item[0])
            for fig_data_category_customdata_item in fig_data_category_customdata
        ]
        # gantt_selected_bars_for_category = np.full(fig_data_category_len, BAR_UNSELECTED, dtype='i4')
        gantt_selected_bars.append(gantt_selected_bars_for_category)
        fig_data_category['marker']['opacity'] = pd.Series(gantt_selected_bars_for_category).map(OPACITY_BY_BAR_SELECTION_STATUS).values

        # fig_data_category['marker']['pattern'] = {'solidity': 0.5, 'shape': '/', 'fillmode': 'overlay', 'fgopacity': 0.5, 'fgcolor': 'rgba(199, 100, 50, 1)', 'bgcolor': 'rgba(199, 100, 50, 1)'}
    fig.update_layout(
        selectionrevision=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )

    # import json
    # fig_dict = json.loads(fig.to_json())
    # print(f'fig={fig_dict}')
    # print(f'gantt_selected_bars={gantt_selected_bars}')

    return fig, selected_datasets_output, gantt_selected_bars,


@callback(
    Output(DATASETS_TABLE_ID, 'columns'),
    Output(DATASETS_TABLE_ID, 'data'),
    Output(DATASETS_TABLE_ID, 'selected_rows'),
    Output(DATASETS_TABLE_ID, 'selected_row_ids'),
    Input(GANTT_SELECTED_DATASETS_IDX_STORE_ID, 'data'),
    Input(DATASETS_TABLE_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    State(DATASETS_STORE_ID, 'data'),
    State(DATASETS_TABLE_ID, 'selected_row_ids'),
    prevent_initial_call=True,
)
@log_exception
def datasets_as_table(
        datasets_indices,
        datasets_table_checklist_all_none_switch,
        datasets_json,
        previously_selected_row_ids
):
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
    if datasets_indices is None:
        datasets_indices = []
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


@_callback_with_exc_handling(
    Output(QUICKLOOK_POPUP_ID, 'children'),
    Input(DATASETS_TABLE_ID, 'active_cell'),
    State(DATASETS_STORE_ID, 'data'),
    prevent_initial_call=True,
)
@log_exception
def popup_graphs(active_cell, datasets_json):
    if datasets_json is None or active_cell is None:
        return []  # children=None instead of [] does not work

    datasets_df = pd.read_json(datasets_json, orient='split', convert_dates=['time_period_start', 'time_period_end'])
    ds_md = datasets_df.loc[active_cell['row_id']]

    if 'selector' in ds_md and isinstance(ds_md['selector'], str) and len(ds_md['selector']) > 0:
        selector = ds_md['selector']
    else:
        selector = None

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
                # due to performance, make a random subsampling of the timeseries
                lambda da: data_processing.utils.subsampling(da.to_series(), n=3000),
                da_by_var
            )
            fig = charts.multi_line(series_by_var) #, width=1000)
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
    Output(SELECT_DATASETS_BUTTON_ID, 'disabled'),
    Input(DATASETS_TABLE_ID, 'selected_row_ids'),
)
@log_exception
def select_datasets_button_disabled(selected_row_ids):
    return not selected_row_ids


@callback(
    Output(INTEGRATE_DATASETS_REQUEST_ID, 'data'),
    Output(APP_TABS_ID, 'value', allow_duplicate=True),
    Input(SELECT_DATASETS_BUTTON_ID, 'n_clicks'),
    State(DATASETS_STORE_ID, 'data'),
    State(DATASETS_TABLE_ID, 'selected_row_ids'),
    prevent_initial_call=True,
)
@log_exception
def select_datasets(n_clicks, datasets_json, selected_row_ids):
    if datasets_json is None or selected_row_ids is None:
        raise dash.exceptions.PreventUpdate

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

    if len(read_dataset_requests) == 0:
        raise dash.exceptions.PreventUpdate  # TODO: instead, show a popup with the warning, that no datasets were found

    req = data_processing.IntegrateDatasetsRequest(read_dataset_requests)
    # TODO: do it asynchronously? will it work with dash/flask? look at options of @app.callback decorator (background=True, ???)
    req.compute()

    return req.to_dict(), FILTER_DATA_TAB_VALUE
