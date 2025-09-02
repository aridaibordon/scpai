import os
import inspect

from scpai.eval import predict
from scpai.spectrum import Signal_H
from scpai.data import get_file_attr
from scpai.model import load_model


def get_scpai_path():
    fname = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(fname))


SCPAI_PATH = get_scpai_path()
SCPAI_DATA_PATH = os.path.join(SCPAI_PATH, "data")
