from os.path import join

from dataclasses import dataclass
from typing import Dict, List

import torch
import numpy as np
import torch.nn as nn

from numpy.typing import NDArray

from scpai.config import MODEL_PATH


class SCPAI_H(nn.Module):
    def __init__(
        self, egrid: NDArray[np.float64], layer_list: List[int], n_outputs: int = 2
    ) -> None:
        super().__init__()

        self.name = "SCPAI_H"
        self.description = "SCPAI for homogeneous analysis"

        nn_sequence, last_layer = [], len(egrid)
        for layer in layer_list:
            nn_sequence.append(nn.Linear(last_layer, layer))
            nn_sequence.append(nn.ReLU())
            last_layer = layer

        self.model = nn.Sequential(
            *nn_sequence,
            nn.Linear(last_layer, n_outputs),
        )

    def forward(self, x):
        return self.model(x)


class SCPAI_MZ(nn.Module):
    def __init__(
        self, egrid: NDArray[np.float64], nzones: int, layer_list: List[int]
    ) -> None:
        super().__init__()

        self.name = "SCPAI_MZ"
        self.description = f"SCPAI for multizone analysis ({nzones} zones)"

        nn_sequence, layer_size = [], len(egrid)
        for layer in layer_list:
            nn_sequence.append(nn.Linear(layer_size, layer))
            nn_sequence.append(nn.ReLU())
            layer_size = layer

        nn_sequence.append(nn.Linear(layer_size, 2 * nzones))
        self.model = nn.Sequential(*nn_sequence)

    def forward(self, x):
        return self.model(x)


@dataclass
class SCPAI_H_Config:
    generator = SCPAI_H
    egrid: NDArray[np.float64]
    layer_list: List[int]
    path: str

    def __post_init__(self) -> None:
        self.path = join(MODEL_PATH, self.path)


@dataclass
class SCPAI_MZ_Config:
    generator = SCPAI_MZ
    egrid: NDArray[np.float64]
    layer_list: List[int]
    nzones: int
    path: str

    def __post_init__(self) -> None:
        self.path = join(MODEL_PATH, self.path)


MODEL_H_DATABASE: Dict[str, SCPAI_H_Config] = {
    "paper_Ar_H000": SCPAI_H_Config(
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f000.pth",
    ),
    "paper_Ar_H005": SCPAI_H_Config(
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f005.pth",
    ),
    "paper_Ar_H010": SCPAI_H_Config(
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f010.pth",
    ),
    "paper_Ar_H015": SCPAI_H_Config(
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f015.pth",
    ),
    "paper_Ar_H020": SCPAI_H_Config(
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f020.pth",
    ),
}

MODEL_MZ_DATABASE: Dict[str, SCPAI_MZ_Config] = {
    "test_Kr_MZ2_010": SCPAI_MZ_Config(
        np.linspace(15200, 15600, 401),
        [1024, 1024, 1024, 1024],
        2,
        "mz/Kr_MZ2_f010.pth",
    ),
    # OMEGA 2024 magnetized
    "omega2024_mag_MZ3": SCPAI_MZ_Config(
        np.linspace(3000, 4000, 2001),
        [1024, 1024, 1024, 1024],
        3,
        "mz/Ar_MZ3_F010_15um_full.pth",
    ),
    "omega2024_mag_MZ4": SCPAI_MZ_Config(
        np.linspace(3000, 4000, 2001),
        [1024, 1024, 1024, 1024],
        4,
        "mz/Ar_MZ4_F010_15um_full.pth",
    ),
    "omega2024_mag_MZ5": SCPAI_MZ_Config(
        np.linspace(3000, 4000, 2001),
        [1024, 1024, 1024, 1024],
        5,
        "mz/Ar_MZ5_F010_15um_full.pth",
    ),
    "omega2024_mag_MZ6": SCPAI_MZ_Config(
        np.linspace(3000, 4000, 2001),
        [1024, 1024, 1024, 1024],
        6,
        "mz/Ar_MZ6_F010_15um_full.pth",
    ),
    # OMEGA 2024 unmagnetized
    "omega2024_unmag_MZ3": SCPAI_MZ_Config(
        np.linspace(3200, 4000, 1601),
        [1024, 1024, 1024, 1024],
        3,
        "mz/Ar_MZ3_F010_15um_nohea.pth",
    ),
    "omega2024_unmag_MZ4": SCPAI_MZ_Config(
        np.linspace(3200, 4000, 1601),
        [1024, 1024, 1024, 1024],
        4,
        "mz/Ar_MZ4_F010_15um_nohea.pth",
    ),
    "omega2024_unmag_MZ5": SCPAI_MZ_Config(
        np.linspace(3200, 4000, 1601),
        [1024, 1024, 1024, 1024],
        5,
        "mz/Ar_MZ5_F010_15um_nohea.pth",
    ),
}


def load_h_model(model_name: str, device: str | None = None) -> SCPAI_H:
    if model_name not in MODEL_H_DATABASE.keys():
        raise KeyError(f"{model_name} is not a valid model.")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = MODEL_H_DATABASE[model_name]
    model_gen, egrid, layer_list, path = (
        model_config.generator,
        model_config.egrid,
        model_config.layer_list,
        model_config.path,
    )

    model = model_gen(egrid, layer_list).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))

    return model


def load_mz_model(model_name: str, device: str | None = None) -> SCPAI_MZ:
    if model_name not in MODEL_MZ_DATABASE.keys():
        raise KeyError(f"{model_name} is not a valid model.")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = MODEL_MZ_DATABASE[model_name]
    model_gen, egrid, layer_list, nzones, path = (
        model_config.generator,
        model_config.egrid,
        model_config.layer_list,
        model_config.nzones,
        model_config.path,
    )

    model = model_gen(egrid, nzones, layer_list).to(device)
    model.load_state_dict(torch.load(path, weights_only=True))

    return model


def load_mz_config(model_name: str) -> SCPAI_MZ_Config:
    if model_name not in MODEL_MZ_DATABASE.keys():
        raise KeyError(f"{model_name} is not a valid model.")

    return MODEL_MZ_DATABASE[model_name]
