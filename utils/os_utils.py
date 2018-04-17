import os
import pickle
import logging

from urllib.request import urlopen


def create_if_not_exists(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def path_join(path, *paths):
    return os.path.join(path, *paths)


def path_exists(path):
    return os.path.exists(path)


def mkdirs(path):
    os.makedirs(path)


def load_obj(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)


def dump_obj(obj, path):
    with open(path, 'wb') as fp:
        return pickle.dump(obj, fp)


def download_small_file(src, dst):
    with urlopen(src) as remote, open(dst, 'wb') as local:
        local.write(remote.read())


DEFAULT_LOGGER = None


def get_default_logger():
    global DEFAULT_LOGGER
    if DEFAULT_LOGGER is None:
        DEFAULT_LOGGER = logging.getLogger('ALL')
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(name)s %(levelname)s %(message)s'))

        DEFAULT_LOGGER.setLevel(logging.DEBUG)
        DEFAULT_LOGGER.addHandler(handler)
    return DEFAULT_LOGGER
