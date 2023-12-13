import logging
from clear_cache import clear_cache, CACHE_DIR
from init_cache import init_cache


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

    clear_cache()
    logging.info(f'{CACHE_DIR} cleared')

    init_cache()
    logging.info(f'{CACHE_DIR} created')
