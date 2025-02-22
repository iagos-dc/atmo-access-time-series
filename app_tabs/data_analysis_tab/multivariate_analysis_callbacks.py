import warnings
import dash
import numpy as np
import pandas as pd
import toolz
from dash import Input

import data_processing
import data_processing.analysis
from app_tabs.common import layout as common_layout
from . import multivariate_analysis_layout, tabs_layout
from data_processing import metadata
from log import logger
from utils import dash_dynamic_components as ddc, charts, helper
from utils.exception_handler import dynamic_callback_with_exc_handling, AppWarning, EmptyFigureException


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ID, 'children'),
    Input(tabs_layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    prevent_initial_call=False,
)
def get_multivariate_analysis_variables_cardbody_callback(tab_id, filter_data_request):
    if filter_data_request is None or tab_id != tabs_layout.MULTIVARIATE_ANALYSIS_TAB_ID:
        raise dash.exceptions.PreventUpdate

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = filter_data_request.compute()
    da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)

    vs = list(metadata_by_var)
    if len(vs) <= 1:
        warnings.warn('For multivariate analysis choose at least 2 variables', category=AppWarning)
        return []

    options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
    options_c = ([{'label': '---', 'value': '---'}] + options) if len(vs) >= 3 else None

    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()

    return [
        multivariate_analysis_layout.get_variable_dropdown(
            dropdown_id=ddc.add_active_to_component_id(dropdown_id),
            axis_label=axis_label,
            options=options,
            value=value,
            disabled=disabled,
            persistence_id=f'{axis_label}:{integrate_datasets_request_hash}'
        )
        for dropdown_id, axis_label, options, value, disabled in zip(
            [
                multivariate_analysis_layout.X_VARIABLE_SELECT_ID,
                multivariate_analysis_layout.Y_VARIABLE_SELECT_ID,
                multivariate_analysis_layout.C_VARIABLE_SELECT_ID
            ],
            ['X', 'Y', 'C'],
            [options, options] + ([options_c] if options_c else []),
            [options[0]['value'], options[1]['value'], '---'],
            [False, False, False],
        )
    ]


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_2_ID, 'children'),
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_3_ID, 'children'),
    Input(tabs_layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
    ddc.DynamicInput(multivariate_analysis_layout.PLOT_TYPE_RADIO_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'disabled'),
    prevent_initial_call=True,
)
def get_extra_parameters(tab_id, plot_type, c_variable, c_variable_disabled):
    if plot_type is None or tab_id != tabs_layout.MULTIVARIATE_ANALYSIS_TAB_ID:
        raise dash.exceptions.PreventUpdate

    if plot_type == multivariate_analysis_layout.INDIVIDUAL_OBSERVATIONS_PLOT:
        return [], []  # children=None instead of [] does not work
    elif plot_type == multivariate_analysis_layout.HEXBIN_PLOT:
        if c_variable == '---' or c_variable_disabled:
            return multivariate_analysis_layout.hexbin_plot_resolution_slider, []  # children=None instead of [] does not work
        else:
            return (
                multivariate_analysis_layout.hexbin_plot_resolution_slider,
                multivariate_analysis_layout.choice_of_aggregators,
            )
    else:
        raise ValueError(f'unknown plot_type={plot_type}')


@dynamic_callback_with_exc_handling(
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID, 'figure'),
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_VARIABLES_CARDHEADER_ID, 'children'),
    Input(tabs_layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(multivariate_analysis_layout.X_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.Y_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'disabled'),
    ddc.DynamicInput(multivariate_analysis_layout.PLOT_TYPE_RADIO_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.HEXBIN_PLOT_RESOLUTION_SLIDER_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.AGGREGATOR_DISPLAY_BUTTONS_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID, 'relayoutData'),
    prevent_initial_call=True,
)
#@log_exectime
def get_multivariate_plot_callback(
        tab_id,
        filter_data_request,
        x_var,
        y_var,
        color_var,
        color_var_disabled,
        plot_type,
        hexbin_resolution,
        agg_func,
        relayout_data,
):
    if tab_id != tabs_layout.MULTIVARIATE_ANALYSIS_TAB_ID:
        raise dash.exceptions.PreventUpdate

    # print(f'relayoutData={relayout_data}')
    dash_ctx = list(dash.ctx.triggered_prop_ids.values())
    # print(f'dash_ctx={dash_ctx}')

    if helper.any_is_None(x_var, y_var, filter_data_request):
        logger().warning(f'prevented update because x_var, y_var, filter_data_request={x_var is None, y_var is None, filter_data_request is None}')
        raise dash.exceptions.PreventUpdate

    # ignore callback fired by relayout_data if it is abount zoom, pan, selectes, etc.
    figure_extent = charts.get_figure_extent(relayout_data)
    # print(f'figure_extent={figure_extent}')

    if dash_ctx == [ddc.add_active_to_component_id(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID)] and not figure_extent:
        logger().warning(f'prevented update with relayout_data={relayout_data}; dash_ctx={dash_ctx}')
        raise dash.exceptions.PreventUpdate

    if color_var == '---' or color_var_disabled:
        color_var = None

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()
    ds = data_processing.MergeDatasetsRequest(filter_data_request).compute()

    selected_vars = [x_var, y_var]
    if color_var is not None:
        selected_vars.append(color_var)
    ds = ds[selected_vars]

    # drop all nan's (take into account only complete observations)
    ds = ds.dropna('time')
    nb_observations = len(ds['time'])
    if nb_observations == 0:
        selected_vars_as_str = ', '.join(selected_vars)
        warnings.warn(
            f'No observations for the variables {selected_vars_as_str}' 
            f'Please select another set of XY- (and C-) variables or change the data filter from the previous step',
            category=AppWarning
        )
        return None, None

    nb_observations_as_str = f'Variables ({nb_observations:.4g} observations)'

    apply_existing_figure_extent = all([
        ddc.add_active_to_component_id(multivariate_analysis_layout.X_VARIABLE_SELECT_ID) not in dash_ctx,
        ddc.add_active_to_component_id(multivariate_analysis_layout.Y_VARIABLE_SELECT_ID) not in dash_ctx,
        common_layout.FILTER_DATA_REQUEST_ID not in dash_ctx,
    ])

    if apply_existing_figure_extent:
        xy_extent_cond, xy_extent_cond_as_str = charts.filter_ds_on_xy_extent(
            ds, figure_extent,
            x_var=x_var, y_var=y_var,
            x_rel_margin=0.1, y_rel_margin=0.1,
        )
        if xy_extent_cond is not True:
            ds_in_figure_extent = ds.where(xy_extent_cond, drop=True)
        else:
            ds_in_figure_extent = ds
    else:
        ds_in_figure_extent = ds
        xy_extent_cond_as_str = None

    X = ds_in_figure_extent[x_var].values
    Y = ds_in_figure_extent[y_var].values
    if color_var is not None:
        C = ds_in_figure_extent[color_var].values
    else:
        C = None

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), ds_in_figure_extent.data_vars)
    # variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    units_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    if plot_type == multivariate_analysis_layout.INDIVIDUAL_OBSERVATIONS_PLOT:
        plot_title = 'Scatter plot'

        if C is not None and len(C) > 0:
            cmin, cmax = np.amin(C), np.amax(C)
        else:
            cmin, cmax = None, None

        if len(X) > 30_000:
            _var_dict = {'X': X, 'Y': Y}
            if C is not None:
                _var_dict['C'] = C
            _df = pd.DataFrame.from_dict(_var_dict).sample(30_000)
            _X = _df['X'].values
            _Y = _df['Y'].values
            _C = _df['C'].values if C is not None else None
        else:
            _X, _Y, _C = X, Y, C
        fig = charts.plotly_scatter2d(
            x=_X, y=_Y, C=_C,
            cmin=cmin, cmax=cmax,
            xaxis_title=f'{x_var} ({units_by_var.get(x_var, "???")})',
            yaxis_title=f'{y_var} ({units_by_var.get(y_var, "???")})',
            colorbar_title=f'{color_var} ({units_by_var.get(color_var, "???")})' if C is not None else None,
            width=1000, height=700,
            marker_size=np.round(np.sqrt(40 * np.log1p(30_000 / len(_X)))) if len(_X) > 0 else 5
        )
    elif plot_type == multivariate_analysis_layout.HEXBIN_PLOT:
        plot_title = 'Hex-bin plot'
        if C is not None and agg_func is None or hexbin_resolution is None:
            return dash.no_update, nb_observations_as_str
        try:
            fig = charts.plotly_hexbin(
                x=X, y=Y, C=C,
                reduce_function=multivariate_analysis_layout.AGGREGATOR_FUNCTIONS.get(agg_func),
                mode='3d+sample_size_as_hexagon_scaling' if C is not None else '2d',
                gridsize=hexbin_resolution,
                min_count=1,
                xaxis_title=f'{x_var} ({units_by_var.get(x_var, "???")})',
                yaxis_title=f'{y_var} ({units_by_var.get(y_var, "???")})',
                colorbar_title=f'{color_var} ({units_by_var.get(color_var, "???")})' if C is not None else 'Sample size',
                width=1000, height=770,
            )
        except EmptyFigureException:
            warnings.warn('No data within this range. Reset the zoom', category=AppWarning)
            return dash.no_update, nb_observations_as_str
    else:
        raise ValueError(f'plot_type={plot_type}')

    if xy_extent_cond_as_str:
        plot_title = f'{plot_title} for {xy_extent_cond_as_str}'
    plot_title = f'{plot_title} from {len(ds_in_figure_extent["time"]):.4g} samples'

    # calculate Pearson and Spearman's rank correlations
    df = ds.to_dataframe()
    pearson_corr = data_processing.analysis.pearson_correlation(df, x_var, y_var)
    spearman_corr = data_processing.analysis.spearman_rank_correlation(df, x_var, y_var)

    plot_title = (f'{plot_title}<br>'
                  f'X-Y correlation from {nb_observations:.4g} observations: Pearson = {pearson_corr:.4g}; '
                  f'Spearman\'s rank = {spearman_corr:.4g}')
    # show title, legend, watermark, etc.
    fig.update_layout(
        legend=dict(orientation='h'),
        title=plot_title,
        margin={'t': 70},
        uirevision=','.join([x_var, y_var, integrate_datasets_request_hash]),
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    fig = charts.add_watermark(fig)

    # print(f'get_plot_callback fig size={len(fig.to_json()) / 1e3}k')

    return fig, nb_observations_as_str
