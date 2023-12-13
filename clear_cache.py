import shutil
import pathlib
import pkg_resources


CACHE_DIR = pathlib.Path(pkg_resources.resource_filename('data_access', 'cache'))


def clear_cache():
    # if CACHE_DIR.exists():
    #   shutil.rmtree(CACHE_DIR)
    (CACHE_DIR / 'stations_iagos.pkl').unlink(missing_ok=True)
    (CACHE_DIR / 'stations_icos.pkl').unlink(missing_ok=True)
    (CACHE_DIR / 'variables_iagos.pkl').unlink(missing_ok=True)
    (CACHE_DIR / 'variables_icos.pkl').unlink(missing_ok=True)
    if (CACHE_DIR / 'cache.tmp').exists():
        shutil.rmtree(CACHE_DIR / 'cache.tmp')


if __name__ == '__main__':
    clear_cache()
