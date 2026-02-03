import os
import inspect

from scpai.config import get_data_path, load_parameter_grid
from scpai.eval import predict_h, predict_mz
from scpai.spectrum import Signal_H, Signal_MZ
from scpai.data import get_file_attr
from scpai.model import load_h_model, load_mz_model, load_mz_config


def get_scpai_path():
    fname = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(fname))


SCPAI_PATH = get_scpai_path()
SCPAI_DATA_PATH = os.path.join(SCPAI_PATH, "data")
