import torch

import numpy as np
import torch.nn as nn

from os.path import join

from scpai.config import MODEL_PATH


class SCPAI_H(nn.Module):
    def __init__(self, egrid: list, layer_list: list, n_outputs: int = 2) -> None:
        super().__init__()

        self.name = "SCPAI_H"
        self.description = "SCPAI for homogeneous analysis"

        nn_sequence, last_layer = [], len(egrid)
        for layer in layer_list:
            nn_sequence.append(nn.Linear(last_layer, layer))
            nn_sequence.append(nn.ReLU())
            last_layer = layer

        nn_sequence.append(nn.Linear(last_layer, n_outputs))

        self.model = nn.Sequential(*nn_sequence)

    def forward(self, x):
        return self.model(x)


class SCPAI_MZ(nn.Module):
    def __init__(self, egrid: list, nzones: int) -> None:
        super().__init__()

        self.name = "SCPAI_MZ"
        self.description = f"SCPAI for multizone analysis ({nzones} zones)"

        self.model = nn.Sequential(
            nn.Linear(len(egrid), 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 3 * nzones),
        )

    def forward(self, x):
        return self.model(x)


def generate_model_description(model_gen, egrid, layer_list, weights_path):
    return {
        "model_gen": model_gen,
        "egrid": egrid,
        "layer_list": layer_list,
        "weights_path": weights_path,
    }


MODEL_DATABASE = {
    "H000_n1": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "H_f000_n1.pth",
    ),
    "H003_n1": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "H_f003_n1.pth",
    ),
    "H003_n2": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512, 512, 512],
        "H_f003_n2.pth",
    ),
}


def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_data = MODEL_DATABASE[model_name]
    model_gen, egrid, layer_list, weights_path = (
        model_data["model_gen"],
        model_data["egrid"],
        model_data["layer_list"],
        model_data["weights_path"],
    )

    model = model_gen(egrid, layer_list).to(device)
    model.load_state_dict(torch.load(join(MODEL_PATH, weights_path), weights_only=True))

    return model
