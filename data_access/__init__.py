import pathlib
import importlib.resources

CACHE_DIR = pathlib.PurePath(importlib.resources.files('data_access') / 'cache')
pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)

from .data_access import (
    get_stations,
    get_vars,
    get_vars_long,
    get_std_ECV_name_by_code,
    get_datasets,
    filter_datasets_on_vars,
    filter_datasets_on_stations,
    read_dataset,
)
