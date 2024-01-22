import functools
import plotly.express as px

import data_access


@functools.lru_cache()
def get_color_by_variable_code_dict():
    std_ECV_names = list(data_access.get_std_ECV_name_by_code().keys())
    _colors = px.colors.qualitative.Dark24
    assert len(std_ECV_names) <= len(_colors)
    return dict(zip(std_ECV_names, _colors))


@functools.lru_cache()
def get_color_by_std_ECV_name_dict():
    std_ECV_names = list(data_access.get_std_ECV_name_by_code().values())
    _colors = px.colors.qualitative.Dark24
    assert len(std_ECV_names) <= len(_colors)
    return dict(zip(std_ECV_names, _colors))
