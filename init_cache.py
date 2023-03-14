if __name__ == '__main__':
    import data_access
    data_access.get_stations()
    data_access.get_vars_long()
    data_access.get_datasets(variables=None, lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)
