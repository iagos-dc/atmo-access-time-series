import pathlib
import pkg_resources


CACHE_DIR = pathlib.PurePath(pkg_resources.resource_filename('data_access', 'cache'))
DATA_DIR = pathlib.PurePath(pkg_resources.resource_filename('data_access', 'resources'))

pathlib.Path(CACHE_DIR).mkdir(exist_ok=True)
