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
    def __init__(self, egrid: list, nzones: int, layer_list: list) -> None:
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


def generate_model_description(model_gen, egrid, layer_list, weights_path):
    return {
        "model_gen": model_gen,
        "egrid": egrid,
        "layer_list": layer_list,
        "weights_path": weights_path,
    }


MODEL_DATABASE = {
    "paper_Ar_H000": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f000.pth",
    ),
    "paper_Ar_H003": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f003.pth",
    ),
    "paper_Ar_H005": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f005.pth",
    ),
    "paper_Ar_H010": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f010.pth",
    ),
    "paper_Ar_H015": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f015.pth",
    ),
    "paper_Ar_H020": generate_model_description(
        SCPAI_H,
        np.linspace(3500, 4300, 800),
        [512, 512, 512],
        "paper/Ar_H_f020.pth",
    ),
}


def load_model(model_name: str, device: str | None = None):
    if model_name not in MODEL_DATABASE.keys():
        raise KeyError(f"{model_name} is not a valid model.")

    if not device:
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
