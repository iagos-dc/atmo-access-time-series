import toolz
import numpy as np
import xarray as xr
import pandas as pd
import dash
from dash import Input, State, Output, callback
import plotly.express as px

from ..common import layout as common_layout
from . import layout
import data_processing
from data_processing import analysis, metadata
from utils import charts, combo_input_AIO, dash_dynamic_components as ddc
from utils.broadcast import broadcast
from log import log_exectime, log_exception



@callback(
    Output(layout.DATA_ANALYSIS_PARAMETERS_CARDBODY_ID, 'children'),
    Output(layout.DATA_ANALYSIS_FIGURE_CONTAINER_ID, 'children'),
    Input(layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
)
@log_exception
def get_parameters_carbody_content(tab_id):
    if tab_id == layout.EXPLORATORY_ANALYSIS_TAB_ID:
        param_cardbody_children = layout.get_exploratory_analysis_parameters()
        figure_container_children = layout.get_exploratory_plot()
    elif tab_id == layout.TREND_ANALYSIS_TAB_ID:
        param_cardbody_children = []
        figure_container_children = []
    elif tab_id == layout.MULTIVARIATE_ANALYSIS_TAB_ID:
        param_cardbody_children = layout.get_multivariate_analysis_parameters()
        figure_container_children = layout.get_multivariate_plot()
    else:
        raise ValueError(f'unknown tab_id={tab_id}')
    return param_cardbody_children, figure_container_children


@ddc.dynamic_callback(
    ddc.DynamicOutput(layout.DATA_ANALYSIS_METHOD_RADIO_ID, 'options'),
    ddc.DynamicOutput(layout.DATA_ANALYSIS_METHOD_RADIO_ID, 'value'),
    Input(layout.KIND_OF_ANALYSIS_TABS_ID, 'active_tab'),
)
@log_exception
def get_analysis_method_options(tab_id):
    analysis_methods = layout.ANALYSIS_METHOD_LABELS_BY_KIND_OF_ANALYSIS_TABS_ID[tab_id]
    options = [
        {'label': analysis_method, 'value': analysis_method}
        for analysis_method in analysis_methods
    ]
    value = analysis_methods[0] if len(analysis_methods) > 0 else None
    return options, value


@ddc.dynamic_callback(
    ddc.DynamicOutput(layout.VARIABLES_CHECKLIST_ID, 'options'),
    ddc.DynamicOutput(layout.VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(layout.VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
    prevent_initial_call=False,
)
@log_exception
def get_variables_callback(filter_data_request, variables_checklist_all_none_switch):
    if filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)

    vs = list(metadata_by_var)
    if len(vs) == 0:
        raise dash.exceptions.PreventUpdate

    options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
    if variables_checklist_all_none_switch:
        value = vs
    else:
        value = []

    return options, value


# @ddc.dynamic_callback(
#     ddc.DynamicOutput(layout.X_VARIABLE_SELECT_ID, 'options'),
#     ddc.DynamicOutput(layout.X_VARIABLE_SELECT_ID, 'value'),
#     ddc.DynamicOutput(layout.Y_VARIABLE_SELECT_ID, 'options'),
#     ddc.DynamicOutput(layout.Y_VARIABLE_SELECT_ID, 'value'),
#     ddc.DynamicOutput(layout.C_VARIABLE_SELECT_ID, 'options'),
#     ddc.DynamicOutput(layout.C_VARIABLE_SELECT_ID, 'value'),
#     Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
#     prevent_initial_call=False,
# )
# @log_exception
# def get_x_y_c_variables_callback(filter_data_request):
#     if filter_data_request is None:
#         raise dash.exceptions.PreventUpdate
#
#     req = data_processing.FilterDataRequest.from_dict(filter_data_request)
#     da_by_var = req.compute()
#     da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
#     metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)
#
#     vs = list(metadata_by_var)
#     if len(vs) <= 1:
#         return (None, ) * 6
#
#     options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
#     options_c = [{'label': '---', 'value': '---'}]
#     if len(vs) >= 3:
#         options_c.extend(options)
#
#     return options, vs[0], options, vs[1], options_c, '---',


@ddc.dynamic_callback(
    ddc.DynamicOutput(layout.MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ID, 'children'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    # prevent_initial_call=False,
)
@log_exception
def get_multivariate_analysis_variables_cardbody_callback(filter_data_request):
    if filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)

    vs = list(metadata_by_var)
    if len(vs) <= 1:
        return layout.get_message_not_enough_variables_for_multivariate_analysis()

    options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
    options_c = ([{'label': '---', 'value': '---'}] + options) if len(vs) >= 3 else None

    return [
        layout.get_variable_dropdown(
            dropdown_id=ddc.add_active_to_component_id(dropdown_id),
            axis_label=axis_label,
            options=options,
        )
        for dropdown_id, axis_label, options in zip(
            [layout.X_VARIABLE_SELECT_ID, layout.Y_VARIABLE_SELECT_ID, layout.C_VARIABLE_SELECT_ID],
            ['X axis', 'Y axis', 'Colour'],
            [options, options] + ([options_c] if options_c else []),
        )
    ]


@ddc.dynamic_callback(
    ddc.DynamicOutput(layout.DATA_ANALYSIS_METHOD_PARAMETERS_CARDBODY_ID, 'children'),
    ddc.DynamicInput(layout.DATA_ANALYSIS_METHOD_RADIO_ID, 'value'),
)
@log_exception
def get_data_analysis_specification_store(analysis_method):
    if analysis_method == layout.GAUSSIAN_MEAN_AND_STD_METHOD:
        return layout.gaussian_mean_and_std_parameters_combo_input
    elif analysis_method == layout.PERCENTILES_METHOD:
        return layout.percentiles_parameters_combo_input
    elif analysis_method == layout.SCATTER_PLOT_METHOD:
        return layout.get_scatter_plot_parameters()
    elif analysis_method == layout.LINEAR_REGRESSION_METHOD:
        return None


@ddc.dynamic_callback(
    ddc.DynamicOutput(layout.EXPLORATORY_GRAPH_ID, 'figure'),
    ddc.DynamicInput(layout.VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    Input(combo_input_AIO.get_combo_input_data_store_id(layout.EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID), 'data'),
    ddc.DynamicInput(layout.GRAPH_SCATTER_MODE_RADIO_ID, 'value'),
    ddc.DynamicState(layout.DATA_ANALYSIS_METHOD_RADIO_ID, 'value'),
    ddc.DynamicState(layout.EXPLORATORY_GRAPH_ID, 'relayoutData'),
    prevent_initial_call=True,
)
@log_exception
#@log_exectime
def get_exploratory_plot_callback(vs, filter_data_request, method_inputs, scatter_mode, analysis_method, relayout_data):
    # # print(f'relayout_data={relayout_data}')
    if any(map(
            lambda obj: obj is None,
            (dash.ctx.triggered_id, vs, filter_data_request, analysis_method, method_inputs)
    )):
        raise dash.exceptions.PreventUpdate

    # # print(f'method_inputs={method_inputs}')

    req = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = req.compute()
    colors_by_var = charts.get_color_mapping(da_by_var)

    da_by_var = toolz.keyfilter(lambda v: v in vs, da_by_var)
    if len(da_by_var) == 0:
        return charts.empty_figure()

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)
    variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    yaxis_label_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    if analysis_method == layout.GAUSSIAN_MEAN_AND_STD_METHOD:
        analysis_spec = method_inputs['value'][layout.GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID]
        aggregation_period = analysis_spec[layout.AGGREGATION_PERIOD_RADIO_ID]
        min_sample_size = analysis_spec[layout.MIN_SAMPLE_SIZE_INPUT_ID]

        get_gaussian_mean_and_std_by_var = broadcast([0])(analysis.gaussian_mean_and_std)
        mean_std_count_by_var = get_gaussian_mean_and_std_by_var(
            da_by_var,
            aggregation_period,
            min_sample_size=min_sample_size
        )

        mean_by_var, std_by_var, _ = (
            toolz.valmap(lambda t: t[i], mean_std_count_by_var)
            for i in range(3)
        )

        graph_controllers = method_inputs['value'][layout.GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID]
        show_std = graph_controllers[layout.SHOW_STD_SWITCH_ID]
        std_mode = graph_controllers[layout.STD_MODE_RADIO_ID]

        _, period_adjective = layout.AGGREGATION_PERIOD_WORDINGS[aggregation_period]
        plot_title = f'{period_adjective} mean'
        if show_std:
            plot_title += ' and standard deviation'

        width = 1200
        fig = charts.multi_line(
            mean_by_var,
            df_std=std_by_var if show_std else None,
            std_mode=std_mode,
            width=width, height=600,
            scatter_mode=scatter_mode,
            variable_label_by_var=variable_label_by_var,
            yaxis_label_by_var=yaxis_label_by_var,
            color_mapping=colors_by_var,
        )
    elif analysis_method == layout.PERCENTILES_METHOD:
        analysis_spec = method_inputs['value'][layout.PERCENTILES_COMBO_INPUT_AIO_ID]
        aggregation_period = analysis_spec[layout.AGGREGATION_PERIOD_RADIO_ID]
        min_sample_size = analysis_spec[layout.MIN_SAMPLE_SIZE_INPUT_ID]
        percentiles = analysis_spec[layout.PERCENTILES_CHECKLIST_ID]
        user_def_percentile = analysis_spec[layout.PERCENTILE_USER_DEF_INPUT_ID]
        if user_def_percentile is not None:
            percentiles = list(percentiles) + [user_def_percentile]
        percentiles = sorted(set(percentiles))

        get_percentiles_by_var = broadcast([0])(analysis.percentiles)
        quantiles_by_p_and_count_by_var = get_percentiles_by_var(
            da_by_var,
            aggregation_period,
            p=percentiles,
            min_sample_size=min_sample_size
        )

        quantiles_by_p_by_var, _ = (
            toolz.valmap(lambda t: t[i], quantiles_by_p_and_count_by_var)
            for i in range(2)
        )

        def percentile_to_str(p):
            if p == 0:
                return 'min'
            elif p == 100:
                return 'max'
            else:
                return str(round(p))

        quantiles_by_p_by_var = toolz.valmap(
            lambda quantiles_by_p: toolz.keymap(percentile_to_str, quantiles_by_p),
            quantiles_by_p_by_var
        )

        _, period_adjective = layout.AGGREGATION_PERIOD_WORDINGS[aggregation_period]
        plot_title = f'{period_adjective} percentiles: ' + ', '.join(map(percentile_to_str, percentiles))

        width = 1200
        fig = charts.multi_line(
            quantiles_by_p_by_var,
            width=width, height=600,
            scatter_mode=scatter_mode,
            variable_label_by_var=variable_label_by_var,
            yaxis_label_by_var=yaxis_label_by_var,
            color_mapping=colors_by_var,
            line_dash_style_by_sublabel=layout.LINE_DASH_STYLE_BY_PERCENTILE,
        )
    else:
        raise dash.exceptions.PreventUpdate
        #raise NotImplementedError(analysis_method)

    # show title, legend, watermark, etc.
    fig.update_layout(
        legend=dict(orientation='h'),
        title=plot_title.capitalize(),
        hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    fig = charts.add_watermark(fig)

    print(f'get_plot_callback fig size={len(fig.to_json()) / 1e3}k')

    if dash.ctx.triggered_id != common_layout.FILTER_DATA_REQUEST_ID:
        # we reset the zoom only if a new filter data request was launched
        fig = charts.apply_figure_extent(fig, relayout_data)
    return fig


@ddc.dynamic_callback(
    ddc.DynamicOutput(layout.MULTIVARIATE_GRAPH_ID, 'figure'),
    ddc.DynamicInput(layout.X_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(layout.Y_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(layout.C_VARIABLE_SELECT_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(layout.DATA_ANALYSIS_METHOD_RADIO_ID, 'value'),
    ddc.DynamicInput(layout.SCATTER_PLOT_PARAMS_RADIO_ID, 'value'),
    ddc.DynamicInput(layout.MULTIVARIATE_GRAPH_ID, 'relayoutData'),
    prevent_initial_call=True,
)
@log_exception
#@log_exectime
def get_multivariate_plot_callback(x_var, y_var, color_var, filter_data_request, analysis_method, scatter_plot_params, relayout_data):
    # # print(f'relayout_data={relayout_data}')
    if any(map(
            lambda obj: obj is None,
            (dash.ctx.triggered_id, x_var, y_var, color_var, filter_data_request, analysis_method)
    )):
        raise dash.exceptions.PreventUpdate

    # # print(f'method_inputs={method_inputs}')

    filter_datasets_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    ds = data_processing.MergeDatasetsRequest(filter_datasets_request).compute()

    selected_vars = [x_var, y_var]
    if color_var != '---':
        selected_vars.append(color_var)
    ds = ds[selected_vars]

    plot_title = 'Hexbin plot'
    #from {len(ds["time"]):.4g} samples'

    if dash.ctx.triggered_id == ddc.add_active_to_component_id(layout.MULTIVARIATE_GRAPH_ID)\
            or dash.ctx.triggered_id == ddc.add_active_to_component_id(layout.SCATTER_PLOT_PARAMS_RADIO_ID):
        # apply x- and y-data filtering according to figure extent (zoom)
        xy_extent = charts.get_figure_extent(relayout_data)
        x_min, x_max = xy_extent.get('xaxis', {}).get('range', [None, None])
        y_min, y_max = xy_extent.get('yaxis', {}).get('range', [None, None])
        xy_extent_cond = True
        xy_extent_cond_as_str = []

        if x_min is not None:
            xy_extent_cond &= (ds[x_var] >= x_min)
        if x_max is not None:
            xy_extent_cond &= (ds[x_var] <= x_max)

        if x_min is not None and x_max is not None:
            xy_extent_cond_as_str.append(f'{x_min:.4g} <= {x_var} <= {x_max:.4g}')
        elif x_min is not None:
            xy_extent_cond_as_str.append(f'{x_min:.4g} <= {x_var}')
        elif x_max is not None:
            xy_extent_cond_as_str.append(f'{x_var} <= {x_max:.4g}')

        if y_min is not None:
            xy_extent_cond &= (ds[y_var] >= y_min)
        if y_max is not None:
            xy_extent_cond &= (ds[y_var] <= y_max)

        if y_min is not None and y_max is not None:
            xy_extent_cond_as_str.append(f'{y_min:.4g} <= {y_var} <= {y_max:.4g}')
        elif y_min is not None:
            xy_extent_cond_as_str.append(f'{y_min:.4g} <= {y_var}')
        elif y_max is not None:
            xy_extent_cond_as_str.append(f'{y_var} <= {y_max:.4g}')

        if xy_extent_cond is not True:
            ds = ds.where(xy_extent_cond)
            xy_extent_cond_as_str = ' and '.join(xy_extent_cond_as_str)
            plot_title = f'{plot_title} for {xy_extent_cond_as_str}'
    ds = ds.dropna('time')

    # colors_by_var = charts.get_color_mapping(ds.data_vars)

    X = ds[x_var].values
    Y = ds[y_var].values
    if color_var != '---':
        C = ds[color_var].values
    else:
        C = None
    # return charts.empty_figure()

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), ds.data_vars)
    # variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    units_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    if analysis_method == layout.SCATTER_PLOT_METHOD:
        if scatter_plot_params == layout.SCATTER_PLOT_PARAM_INDIVIDUAL_OBSERVATIONS:
            return charts.empty_figure()
        elif scatter_plot_params == layout.SCATTER_PLOT_PARAM_HEXBIN:
            fig = charts.plotly_hexbin(
                x=X, y=Y, C=C,
                reduce_function=np.mean,
                # reduce_function=lambda a: np.quantile(a, 0.75),
                mode='3d+sample_size_as_hexagon_scaling' if C is not None else '2d',
                gridsize=20,
                min_count=1,
                xaxis_title=f'{x_var} ({units_by_var.get(x_var, "???")})',
                yaxis_title=f'{y_var} ({units_by_var.get(y_var, "???")})',
                colorbar_title=f'{color_var} ({units_by_var.get(color_var, "???")})' if C is not None else 'Sample size',
                width=1000, height=700,
            )
        else:
            raise ValueError(f'scatter_plot_params={scatter_plot_params}')
    elif analysis_method == layout.LINEAR_REGRESSION_METHOD:
        return charts.empty_figure()
    else:
        raise ValueError(f'analysis_method={analysis_method}')

    plot_title = f'{plot_title} from {len(ds["time"]):.4g} samples'
    # show title, legend, watermark, etc.
    fig.update_layout(
        legend=dict(orientation='h'),
        title=plot_title.capitalize(),
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    fig = charts.add_watermark(fig)

    print(f'get_plot_callback fig size={len(fig.to_json()) / 1e3}k')

    if dash.ctx.triggered_id == ddc.add_active_to_component_id(layout.SCATTER_PLOT_PARAMS_RADIO_ID):
        # we keep the zoom only if a scatter plot parameters have changed
        fig = charts.apply_figure_extent(fig, relayout_data)
    return fig
