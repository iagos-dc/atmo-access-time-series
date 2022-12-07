import shutil
import pathlib
import pkg_resources


CACHE_DIR = pathlib.PurePath(pkg_resources.resource_filename('data_access', 'cache'))


if __name__ == '__main__':
    shutil.rmtree(CACHE_DIR)
