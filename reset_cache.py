import logging

import config
from clear_cache import clear_cache
from init_cache import init_cache


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    clear_cache()
    logging.info(f'{config.APP_CACHE_DIR} cleared')

    init_cache()
    logging.info(f'{config.APP_CACHE_DIR} created')
