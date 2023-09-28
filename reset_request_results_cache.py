import shutil
import logging
import pathlib
import pkg_resources
import data_access


CACHE_DIR = pathlib.PurePath(pkg_resources.resource_filename('data_access', 'cache'))
CACHE_URL = str(CACHE_DIR / 'cache.tmp')


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    if pathlib.Path(CACHE_URL).exists():
        shutil.rmtree(CACHE_URL)
        logging.info(f'{CACHE_URL} deleted')

    data_access.get_datasets(variables=None, lon_min=-180, lon_max=180, lat_min=-90, lat_max=90)
    logging.info(f'{CACHE_URL} created')
