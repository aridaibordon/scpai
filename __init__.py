import os, inspect

from spectrum import Signal_H
from scpai.data import get_file_attr
from scpai.model import SCPAI_H, SCPAI_H2


def get_scpai_path():
    fname = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(fname))


SCPAI_PATH = get_scpai_path()
SCPAI_DATA_PATH = os.path.join(SCPAI_PATH, "data")
