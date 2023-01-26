import dash
import numpy as np
import pandas as pd
import toolz
from dash import Input

import data_processing
from app_tabs.common import layout as common_layout
from app_tabs.data_analysis_tab import multivariate_analysis_layout, layout
from data_processing import metadata
from log import log_exception
from utils import dash_dynamic_components as ddc, charts


@ddc.dynamic_callback(
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ROW_1_ID, 'children'),
    ddc.DynamicInput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_METHOD_RADIO_ID, 'value'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    prevent_initial_call=False,
)
@log_exception
def get_multivariate_analysis_variables_cardbody_callback(analysis_method, filter_data_request):
    if filter_data_request is None:
        raise dash.exceptions.PreventUpdate

    filter_data_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    da_by_var = filter_data_request.compute()
    da_by_var = {v: da_by_var[v] for v in sorted(da_by_var)}
    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), da_by_var)

    vs = list(metadata_by_var)
    if len(vs) <= 1:
        return multivariate_analysis_layout.get_message_not_enough_variables_for_multivariate_analysis()

    options = [{'label': f'{v} : {md[metadata.VARIABLE_LABEL]}', 'value': v} for v, md in metadata_by_var.items()]
    options_c = ([{'label': '---', 'value': '---'}] + options) if len(vs) >= 3 else None

    integrate_datasets_request_hash = filter_data_request.integrate_datasets_request.deterministic_hash()

    disable_c_select = analysis_method == multivariate_analysis_layout.LINEAR_REGRESSION_METHOD

    return [
        layout.get_variable_dropdown(
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
            ['X axis', 'Y axis', 'Colour'],
            [options, options] + ([options_c] if options_c else []),
            [options[0]['value'], options[1]['value'], '---'],
            [False, False, disable_c_select],
        )
    ]


@ddc.dynamic_callback(
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_2_ID, 'children'),
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_3_ID, 'children'),
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_PARAMETERS_FORM_ROW_4_ID, 'children'),
    ddc.DynamicInput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_METHOD_RADIO_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.PLOT_TYPE_RADIO_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'value'),
)
def get_extra_parameters(analysis_method, plot_type, c_variable):
    if analysis_method != multivariate_analysis_layout.SCATTER_PLOT_METHOD or plot_type != multivariate_analysis_layout.HEXBIN_PLOT:
        return None, None, None

    if c_variable == '---':
        return multivariate_analysis_layout.hexbin_plot_resolution_slider, None, None
    else:
        return (
            multivariate_analysis_layout.hexbin_plot_resolution_slider,
            multivariate_analysis_layout.get_choice_of_aggregators(c_variable),
            multivariate_analysis_layout.aggregator_display_buttons_form,
        )


@ddc.dynamic_callback(
    ddc.DynamicOutput(multivariate_analysis_layout.AGGREGATOR_DISPLAY_BUTTONS_FORM_ID, 'children'),
    ddc.DynamicInput(multivariate_analysis_layout.AGGREGATORS_CHECKLIST_ID, 'value'),
    ddc.DynamicState(multivariate_analysis_layout.AGGREGATOR_DISPLAY_BUTTONS_ID, 'value'),
)
def display_aggregator_display_buttons(agg_options, displayed_agg):
    if agg_options is None:
        raise dash.exceptions.PreventUpdate

    buttons = [
        {'label': agg_option, 'value': agg_option, 'disabled': agg_option not in agg_options}
        for agg_option in multivariate_analysis_layout.AGGREGATOR_CHECKLIST_OPTIONS
    ]
    if displayed_agg in agg_options:
        pressed_button = displayed_agg
    else:
        if len(agg_options) > 0:
            pressed_button = agg_options[-1]
        else:
            pressed_button = None

    return multivariate_analysis_layout.get_aggregator_display_buttons(buttons, pressed_button)


@ddc.dynamic_callback(
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID, 'figure'),
    ddc.DynamicOutput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_VARIABLES_CARDBODY_ROW_2_ID, 'children'),
    ddc.DynamicInput(multivariate_analysis_layout.X_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.Y_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.C_VARIABLE_SELECT_ID, 'disabled'),
    Input(common_layout.FILTER_DATA_REQUEST_ID, 'data'),
    ddc.DynamicInput(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_METHOD_RADIO_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.PLOT_TYPE_RADIO_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.HEXBIN_PLOT_RESOLUTION_SLIDER_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.AGGREGATOR_DISPLAY_BUTTONS_ID, 'value'),
    ddc.DynamicInput(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID, 'relayoutData'),
)
@log_exception
#@log_exectime
def get_multivariate_plot_callback(
        x_var, y_var,
        color_var, color_var_disabled,
        filter_data_request,
        analysis_method,
        scatter_plot_params,
        hexbin_resolution,
        agg_func,
        relayout_data
):
    print(f'analysis_method={analysis_method}, scatter_plot_params={scatter_plot_params}')
    dash_ctx = list(dash.ctx.triggered_prop_ids.values())

    # ignore callback fired by relayout_data if it is abount zoom, pan, selectes, etc.
    figure_extent = charts.get_figure_extent(relayout_data)
    if dash_ctx == [ddc.add_active_to_component_id(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID)] and not figure_extent:
        print(f'prevented update with relayout_data={relayout_data}; dash_ctx={dash_ctx}')
        raise dash.exceptions.PreventUpdate

    #if not dash_ctx or any(
    if any(
            map(
                lambda obj: obj is None,
                (x_var, y_var, filter_data_request, analysis_method)
            )
    ):
        print(f'ulala: {dash_ctx}')
        return dash.no_update, dash.no_update

    if color_var == '---' or color_var_disabled:
        color_var = None

    filter_datasets_request = data_processing.FilterDataRequest.from_dict(filter_data_request)
    ds = data_processing.MergeDatasetsRequest(filter_datasets_request).compute()

    selected_vars = [x_var, y_var]
    if color_var is not None:
        selected_vars.append(color_var)
    ds = ds[selected_vars]

    # drop all nan's (take into account only complete observations)
    ds = ds.dropna('time')
    nb_observations = len(ds['time'])
    nb_observations_as_str = f'# observations = {nb_observations:.4g}'

    apply_existing_figure_extent = all([
        ddc.add_active_to_component_id(multivariate_analysis_layout.X_VARIABLE_SELECT_ID) not in dash_ctx,
        ddc.add_active_to_component_id(multivariate_analysis_layout.Y_VARIABLE_SELECT_ID) not in dash_ctx,
        common_layout.FILTER_DATA_REQUEST_ID not in dash_ctx,
        ddc.add_active_to_component_id(multivariate_analysis_layout.MULTIVARIATE_ANALYSIS_METHOD_RADIO_ID) not in dash_ctx,
    ])

    xy_extent_cond_as_str = None
    if apply_existing_figure_extent:
        # apply x- and y-data filtering according to figure extent (zoom)
        xy_extent = figure_extent if isinstance(figure_extent, dict) else {}
        if xy_extent is not None:
            x_min, x_max = xy_extent.get('xaxis', {}).get('range', [None, None])
            y_min, y_max = xy_extent.get('yaxis', {}).get('range', [None, None])
            xy_extent_cond = True
            xy_extent_cond_as_str = []

            if x_min is not None:
                xy_extent_cond &= (ds[x_var] >= x_min)
            if x_max is not None:
                xy_extent_cond &= (ds[x_var] <= x_max)

            if x_min is not None and x_max is not None:
                xy_extent_cond_as_str.append(f'{x_min:.4g} <= X <= {x_max:.4g}')
            elif x_min is not None:
                xy_extent_cond_as_str.append(f'{x_min:.4g} <= {x_var}')
            elif x_max is not None:
                xy_extent_cond_as_str.append(f'{x_var} <= {x_max:.4g}')

            if y_min is not None:
                xy_extent_cond &= (ds[y_var] >= y_min)
            if y_max is not None:
                xy_extent_cond &= (ds[y_var] <= y_max)

            if y_min is not None and y_max is not None:
                xy_extent_cond_as_str.append(f'{y_min:.4g} <= Y <= {y_max:.4g}')
            elif y_min is not None:
                xy_extent_cond_as_str.append(f'{y_min:.4g} <= {y_var}')
            elif y_max is not None:
                xy_extent_cond_as_str.append(f'{y_var} <= {y_max:.4g}')

            if xy_extent_cond is not True:
                ds = ds.where(xy_extent_cond, drop=True)
                xy_extent_cond_as_str = ' and '.join(xy_extent_cond_as_str)

    X = ds[x_var].values
    Y = ds[y_var].values
    if color_var is not None:
        C = ds[color_var].values
    else:
        C = None

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), ds.data_vars)
    # variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    units_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    if analysis_method == multivariate_analysis_layout.SCATTER_PLOT_METHOD:
        if scatter_plot_params == multivariate_analysis_layout.INDIVIDUAL_OBSERVATIONS_PLOT:
            print('got into')
            plot_type = 'Scatter plot'

            if C is not None and len(C) > 0:
                cmin, cmax = np.amin(C), np.amax(C)
            else:
                cmin, cmax = None, None

            if len(X) > 10_000:
                _var_dict = {'X': X, 'Y': Y}
                if C is not None:
                    _var_dict['C'] = C
                _df = pd.DataFrame.from_dict(_var_dict).sample(10_000)
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
            )
            print('got out')
        elif scatter_plot_params == multivariate_analysis_layout.HEXBIN_PLOT:
            plot_type = 'Hex-bin plot'
            if agg_func is None or hexbin_resolution is None:
                return dash.no_update, nb_observations_as_str
            fig = charts.plotly_hexbin(
                x=X, y=Y, C=C,
                reduce_function=multivariate_analysis_layout.AGGREGATOR_FUNCTIONS.get(agg_func),
                mode='3d+sample_size_as_hexagon_scaling' if C is not None else '2d',
                gridsize=hexbin_resolution,
                min_count=1,
                xaxis_title=f'{x_var} ({units_by_var.get(x_var, "???")})',
                yaxis_title=f'{y_var} ({units_by_var.get(y_var, "???")})',
                colorbar_title=f'{color_var} ({units_by_var.get(color_var, "???")})' if C is not None else 'Sample size',
                width=1000, height=700,
            )
        else:
            raise ValueError(f'scatter_plot_params={scatter_plot_params}')
    elif analysis_method == multivariate_analysis_layout.LINEAR_REGRESSION_METHOD:
        plot_type = 'Linear regression'
        return charts.empty_figure(), nb_observations_as_str
    else:
        raise ValueError(f'analysis_method={analysis_method}')

    plot_title = f'{plot_type}'
    if xy_extent_cond_as_str:
        plot_title = f'{plot_title} for {xy_extent_cond_as_str}'
    plot_title = f'{plot_title} from {len(ds["time"]):.4g} samples'

    # show title, legend, watermark, etc.
    fig.update_layout(
        legend=dict(orientation='h'),
        title=plot_title,
        # hovermode='x',  # performance improvement??? see: https://github.com/plotly/plotly.js/issues/6230
    )
    fig = charts.add_watermark(fig)

    #if ddc.add_active_to_component_id(multivariate_analysis_layout.PLOT_TYPE_RADIO_ID) in dash_ctx or \
    #        ddc.add_active_to_component_id(multivariate_analysis_layout.MULTIVARIATE_GRAPH_ID) in dash_ctx:
        # we keep the zoom only if a scatter plot parameters have changed or the zoom has changed
    if apply_existing_figure_extent:
        fig = charts.apply_figure_extent(fig, relayout_data)

    # print(f'get_plot_callback fig size={len(fig.to_json()) / 1e3}k')
    print(fig)
    return fig, nb_observations_as_str
