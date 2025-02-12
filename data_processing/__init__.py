from .request_manager import (
    request_from_dict,
    request_from_json,
    GetICOSDatasetTitleRequest,
    ReadDataRequest,
    MergeDatasetsRequest,
    IntegrateDatasetsRequest,
    FilterDataRequest,
    requests_deque,
)

from .filtering import (
    interpolate_mask_1d,
    filter_dataset,
)


# could be used in notebooks
def retrieve_and_integrate_datasets(datasets_df):
    read_dataset_requests = []
    for idx, ds_metadata in datasets_df.iterrows():
        ri = ds_metadata['RI']
        url = ds_metadata['url']
        selector = ds_metadata['selector'] if 'selector' in ds_metadata else None
        req = ReadDataRequest(ri, url, ds_metadata, selector=selector)
        read_dataset_requests.append(req)

    req = IntegrateDatasetsRequest(read_dataset_requests)
    return req.compute()
