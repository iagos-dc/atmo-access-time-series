import functools
import plotly.express as px

import data_access


@functools.lru_cache()
def get_color_by_variable_code_dict():
    var_codes = data_access.ECV_by_var_codes.index
    _colors = px.colors.qualitative.Dark24
    assert len(var_codes) <= len(_colors)
    return dict(zip(var_codes, _colors))


@functools.lru_cache()
def get_color_by_ECV_name_dict():
    ecvs = data_access.var_codes_by_ECV.index
    _colors = px.colors.qualitative.Dark24
    assert len(ecvs) <= len(_colors)
    return dict(zip(ecvs, _colors))
