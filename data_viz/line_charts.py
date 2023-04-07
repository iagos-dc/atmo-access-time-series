import toolz
from data_processing import metadata
from utils import charts


LINE_DASH_STYLE_BY_PERCENTILE = {
    'min': 'dot',
    '5': 'dash',
    '25': 'dashdot',
    '50': 'solid',
    '75': 'dashdot',
    '95': 'dash',
    'max': 'dot',
    'other': 'longdashdot'
}


def mean_std_plot(ds_by_var, show_std=True, std_mode='fill', scatter_mode='lines', plot_title=None):
    colors_by_var = charts.get_color_mapping(ds_by_var)

    metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), ds_by_var)
    metadata_by_var = {v: md[f'{v}_mean'] for v, md in metadata_by_var.items()}
    variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    yaxis_label_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    mean_by_var = {v: ds[f'{v}_mean'].to_series() for v, ds in ds_by_var.items()}
    std_by_var = {v: ds[f'{v}_std'].to_series() for v, ds in ds_by_var.items()}

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

    if plot_title:
        fig.update_layout(
            title=plot_title
        )

    fig.update_layout(
        xaxis={'title': 'time'},
    )
    fig = charts.add_watermark(fig)

    return fig


def percentiles_plot(ds_by_var, scatter_mode='lines', plot_title=None):
    colors_by_var = charts.get_color_mapping(ds_by_var)

    _metadata_by_var = toolz.valmap(lambda da: metadata.da_attr_to_metadata_dict(da=da), ds_by_var)
    metadata_by_var = {}
    for v, md in _metadata_by_var.items():
        any_v_p = toolz.first(v_p for v_p in md if v_p.startswith(f'{v}_p'))
        metadata_by_var[v] = md[any_v_p]
    variable_label_by_var = toolz.valmap(lambda md: md[metadata.VARIABLE_LABEL], metadata_by_var)
    yaxis_label_by_var = toolz.valmap(lambda md: md[metadata.YAXIS_LABEL], metadata_by_var)

    def ds_to_series_by_p(ds):
        def percentile_to_str(p):
            if p == 0:
                return 'min'
            elif p == 100:
                return 'max'
            else:
                return str(round(p))

        res = {}
        for v_p, da in ds.data_vars.items():
            *v, p = v_p.split('_')
            v = '_'.join(v)
            p = float(p[1:])
            res[percentile_to_str(p)] = da.to_series()

        return res

    quantiles_by_p_by_var = toolz.valmap(ds_to_series_by_p, ds_by_var)

    width = 1200
    fig = charts.multi_line(
        quantiles_by_p_by_var,
        width=width, height=600,
        scatter_mode=scatter_mode,
        variable_label_by_var=variable_label_by_var,
        yaxis_label_by_var=yaxis_label_by_var,
        color_mapping=colors_by_var,
        line_dash_style_by_sublabel=LINE_DASH_STYLE_BY_PERCENTILE,
    )

    if plot_title:
        fig.update_layout(
            title=plot_title
        )

    fig.update_layout(
        xaxis={'title': 'time'},
    )
    fig = charts.add_watermark(fig)

    return fig
