import dash
import toolz
from dash import Input

import app_tabs.data_analysis_tab.exploratory_analysis_layout
import data_processing
from data_processing import metadata, analysis
from app_tabs.common import layout as common_layout
from app_tabs.data_analysis_tab import layout, multivariate_analysis_layout, exploratory_analysis_layout
from log import log_exception
from utils import dash_dynamic_components as ddc, combo_input_AIO, charts
from utils.broadcast import broadcast


@ddc.dynamic_callback(
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ID, 'options'),
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ALL_NONE_SWITCH_ID, 'value'),
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


@ddc.dynamic_callback(
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_1_ID, 'children'),
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_2_ID, 'children'),
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_3_ID, 'children'),
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_PARAMETERS_FORM_ROW_4_ID, 'children'),
    ddc.DynamicInput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_METHOD_RADIO_ID, 'value'),
)
@log_exception
def get_data_analysis_specification_store(analysis_method):
    if analysis_method == layout.GAUSSIAN_MEAN_AND_STD_METHOD:
        return layout.gaussian_mean_and_std_parameters_combo_input
    elif analysis_method == layout.PERCENTILES_METHOD:
        return layout.percentiles_parameters_combo_input
    elif analysis_method == multivariate_analysis_layout.SCATTER_PLOT_METHOD:
        raise RuntimeError(f'analysis_method={multivariate_analysis_layout.SCATTER_PLOT_METHOD}')
    elif analysis_method == multivariate_analysis_layout.LINEAR_REGRESSION_METHOD:
        return None


@ddc.dynamic_callback(
    ddc.DynamicOutput(exploratory_analysis_layout.EXPLORATORY_GRAPH_ID, 'figure'),
    ddc.DynamicInput(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_VARIABLES_CHECKLIST_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    Input(combo_input_AIO.get_combo_input_data_store_id(
        app_tabs.data_analysis_tab.exploratory_analysis_layout.EXPLORATORY_ANALYSIS_INPUTS_GROUP_ID), 'data'),
    ddc.DynamicInput(exploratory_analysis_layout.EXPLORATORY_GRAPH_SCATTER_MODE_RADIO_ID, 'value'),
    ddc.DynamicState(exploratory_analysis_layout.EXPLORATORY_ANALYSIS_METHOD_RADIO_ID, 'value'),
    ddc.DynamicState(exploratory_analysis_layout.EXPLORATORY_GRAPH_ID, 'relayoutData'),
    prevent_initial_call=True,
)
@log_exception
def get_exploratory_plot_callback(vs, filter_data_request, method_inputs, scatter_mode, analysis_method, relayout_data):
    if any(map(
            lambda obj: obj is None,
            (dash.ctx.triggered_id, vs, filter_data_request, analysis_method, method_inputs)
    )):
        raise dash.exceptions.PreventUpdate

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
        analysis_spec = method_inputs['value'][
            app_tabs.data_analysis_tab.exploratory_analysis_layout.GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID]
        aggregation_period = analysis_spec[
            app_tabs.data_analysis_tab.exploratory_analysis_layout.AGGREGATION_PERIOD_RADIO_ID]
        min_sample_size = analysis_spec[app_tabs.data_analysis_tab.exploratory_analysis_layout.MIN_SAMPLE_SIZE_INPUT_ID]

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

        graph_controllers = method_inputs['value'][
            app_tabs.data_analysis_tab.exploratory_analysis_layout.GAUSSIAN_MEAN_AND_STD_COMBO_INPUT_AIO_ID]
        show_std = graph_controllers[app_tabs.data_analysis_tab.exploratory_analysis_layout.SHOW_STD_SWITCH_ID]
        std_mode = graph_controllers[app_tabs.data_analysis_tab.exploratory_analysis_layout.STD_MODE_RADIO_ID]

        _, period_adjective = app_tabs.data_analysis_tab.exploratory_analysis_layout.AGGREGATION_PERIOD_WORDINGS[aggregation_period]
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
        aggregation_period = analysis_spec[
            app_tabs.data_analysis_tab.exploratory_analysis_layout.AGGREGATION_PERIOD_RADIO_ID]
        min_sample_size = analysis_spec[app_tabs.data_analysis_tab.exploratory_analysis_layout.MIN_SAMPLE_SIZE_INPUT_ID]
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

        _, period_adjective = app_tabs.data_analysis_tab.exploratory_analysis_layout.AGGREGATION_PERIOD_WORDINGS[aggregation_period]
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

    if dash.ctx.triggered_id != common_layout.FILTER_DATA_REQUEST_ID:
        # we reset the zoom only if a new filter data request was launched
        fig = charts.apply_figure_extent(fig, relayout_data)

    # print(f'get_plot_callback fig size={len(fig.to_json()) / 1e3}k')
    return fig