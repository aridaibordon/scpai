import os
import inspect

import numpy as np

from scpai.config import get_data_path
from scpai.eval import predict
from scpai.spectrum import Signal_H
from scpai.data import get_file_attr
from scpai.model import load_model


def get_scpai_path():
    fname = inspect.getframeinfo(inspect.currentframe()).filename
    return os.path.dirname(os.path.abspath(fname))


def load_parameter_grid(specie: str):
    return (
        np.loadtxt(os.path.join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
        np.loadtxt(os.path.join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
    )


SCPAI_PATH = get_scpai_path()
SCPAI_DATA_PATH = os.path.join(SCPAI_PATH, "data")
