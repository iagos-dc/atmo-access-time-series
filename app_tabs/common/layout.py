from dash import dcc

DATASETS_STORE_ID = 'datasets-store'
# 'data' stores datasets metadata in JSON, as provided by the method
# pd.DataFrame.to_json(orient='split', date_format='iso')

INTEGRATE_DATASETS_REQUEST_ID = 'integrate-datasets-request'
# 'data' stores a JSON representation of a request executed

FILTER_DATA_REQUEST_ID = 'filter-data-request'


def get_app_data_stores():
    # these are special Dash components used for transferring data from one callback to other callback(s)
    # without displaying the data
    return [
        dcc.Store(id=DATASETS_STORE_ID, storage_type='session'),
        dcc.Store(id=INTEGRATE_DATASETS_REQUEST_ID, storage_type='session'),
        dcc.Store(id=FILTER_DATA_REQUEST_ID, storage_type='session'),
    ]

# Initialization of global objects
