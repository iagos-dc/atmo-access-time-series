import pathlib
import logging

import config
import data_access
from atmoaccess_data_access import query_actris


logging.basicConfig(format='%(asctime)s - %(levelname)s - aats:init_cache - %(message)s', level=logging.INFO)


def init_cache():
    query_actris._create_cache_and_close(data_access.CACHE_DIR / 'actris-cache.tmp')

    data_access.get_stations()
    data_access.get_datasets()


if __name__ == '__main__':
    init_cache()
