import shutil
import pathlib

import config


def clear_cache():
    if pathlib.Path(config.APP_CACHE_DIR).exists():
        shutil.rmtree(config.APP_CACHE_DIR)


if __name__ == '__main__':
    clear_cache()
