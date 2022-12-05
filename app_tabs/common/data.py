import data_access


def _get_station_by_shortnameRI(stations):
    df = stations.set_index('short_name_RI')[['long_name', 'RI']]
    df['station_fullname'] = df['long_name'] + ' (' + df['RI'] + ')'
    return df


stations = data_access.get_stations()
station_by_shortnameRI = _get_station_by_shortnameRI(stations)
