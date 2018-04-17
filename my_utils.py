from time import time
from utils.tf_utils import *
from utils.os_utils import *

from configs.configuration import Config
from configs.arguments import get_default_parser


class Timer:
    def __enter__(self):
        self._tic = time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.eclipsed = time() - self._tic


def load_config():
    parser = get_default_parser()
    args = parser.parse_args()
    config = Config(args.__dict__.copy())
    return config

