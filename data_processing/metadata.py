# TODO: use this module in app_tabs.filter_data_tab

VARIABLE_LABEL = 'variable_label'
UNITS = 'units'
YAXIS_LABEL = 'yaxis_label'
CITY_OR_STATION_NAME = 'city_or_station_name'


def dict_get(d, *keys, default=None):
    for key in keys:
        try:
            return d[key]
        except KeyError:
            pass
    return default


def da_attr_to_metadata_dict(da=None, attrs=None):
    if da is None and attrs is None:
        raise ValueError('at least da or attrs must be not None')
    if da is not None:
        _attrs = da.attrs.copy()
        _attrs['_da_name'] = da.name
        if attrs is not None:
            _attrs.update(attrs)
        attrs = _attrs

    variable_label = dict_get(attrs, 'long_name', 'label', '_da_name', '??? unknown variable ???')
    units = attrs.get('units', '???')
    yaxis_label = units  # units = f'{variable_label} ({units})'
    city_or_station_name = dict_get(attrs, 'city', 'station_name')

    metadata = {
        VARIABLE_LABEL: variable_label,
        UNITS: units,
        YAXIS_LABEL: yaxis_label,
        CITY_OR_STATION_NAME: city_or_station_name,
    }
    # # print(f'da_metadata={metadata}')
    return metadata
