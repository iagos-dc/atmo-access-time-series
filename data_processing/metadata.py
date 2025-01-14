# TODO: use this module in app_tabs.filter_data_tab
import xarray as xr


VARIABLE_LABEL = 'variable_label'
UNITS = 'units'
YAXIS_LABEL = 'yaxis_label'
CITY_OR_STATION_NAME = 'city_or_station_name'
STATION_CODE = 'station_code'


def dict_get(d, *keys, default=None):
    for key in keys:
        try:
            return d[key]
        except KeyError:
            pass
    return default


def da_attr_to_metadata_dict(da=None, attrs=None):
    if isinstance(da, xr.Dataset):
        metadata_by_var = {}
        for v in da:
            metadata_by_var[v] = da_attr_to_metadata_dict(da[v], attrs=da.attrs)
        return metadata_by_var

    if da is None and attrs is None:
        raise ValueError('at least da or attrs must be not None')
    if da is not None:
        _attrs = da.attrs.copy()
        _attrs['_da_name'] = da.name
        if attrs is not None:
            _attrs.update(attrs)
        attrs = _attrs

    variable_label = dict_get(attrs, 'long_name', 'label', 'standard_name', '_da_name', default='??? unknown variable ???')
    units = attrs.get('units', '???')
    yaxis_label = units  # units = f'{variable_label} ({units})'
    city_or_station_name = dict_get(attrs, 'city', 'station_name', 'region', 'ebas_station_name')
    station_code = dict_get(
        attrs,
        'IATA_code', 'region_code', # IAGOS
        'station_id', # ICOS
        'ebas_station_code', # ACTRIS
    )

    metadata = {
        VARIABLE_LABEL: variable_label,
        UNITS: units,
        YAXIS_LABEL: yaxis_label,
        CITY_OR_STATION_NAME: city_or_station_name,
        STATION_CODE: station_code
    }
    # # print(f'da_metadata={metadata}')
    return metadata
