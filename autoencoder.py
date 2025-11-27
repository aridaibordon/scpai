import os
import torch

from dataclasses import dataclass
from os.path import join
from typing import Dict, List

import numpy as np
import torch.nn as nn
import torch.optim as optin

from numpy.typing import NDArray
from torch.utils.data import Dataset

from scpai.config import LEARNING_RATE, MODEL_PATH, get_data_path
from scpai.spectrum import Signal_H
from scpai.train import train_loop


class Dataset_H_Autoencoder(Dataset):
    def __init__(self, specie: str, signal: Signal_H, mode: str) -> None:
        allowed_modes = ["train", "test"]
        if mode not in allowed_modes:
            raise ValueError(f"{mode} is not a valid mode.")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path = join(get_data_path(specie), mode)

        self.data_Te, self.data_dne = (
            np.loadtxt(join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
            np.loadtxt(join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
        )

        self.signal = signal
        self.specie = specie

    def __len__(self) -> int:
        return len(os.listdir(self.path))

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        fname = np.sort(os.listdir(self.path))[index]

        clength = 30e-4
        nsignal = self.signal.get_nsignal(fname=join(self.path, fname), clength=clength)

        return (
            torch.Tensor(nsignal).to(self.device),
            torch.Tensor(nsignal).to(self.device),
        )


class SpectraAutoencoder(nn.Module):
    def __init__(self, egrid: NDArray[np.int_], layer_list: list) -> None:
        super().__init__()
        self.egrid = egrid

        encoder_sequence = [nn.Linear(len(egrid), layer_list[0]), nn.ReLU()]
        for ind, next in zip(layer_list[:-1], layer_list[1:]):
            encoder_sequence.append(nn.Linear(ind, next))
            encoder_sequence.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_sequence)

        decoder_sequence = []
        for ind, next in zip(layer_list[::-1][:-1], layer_list[::-1][1:]):
            decoder_sequence.append(nn.Linear(ind, next))
            decoder_sequence.append(nn.ReLU())

        self.decoder = nn.Sequential(
            *decoder_sequence,
            nn.Linear(layer_list[0], len(egrid)),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


@dataclass
class AutoencoderConfig:
    generator: SpectraAutoencoder
    egrid: NDArray[np.float64]
    layer_list: List[int]
    path: str

    def __post_init__(self):
        self.path = join(MODEL_PATH, self.path)


AUTOENCODER_DATABASE: Dict[str, AutoencoderConfig] = {
    "test_32": AutoencoderConfig(
        SpectraAutoencoder,
        np.linspace(3500, 4300, 801),
        [128, 64, 32],
        "autoencoder/test_32.pth",
    ),
    "test_8": AutoencoderConfig(
        SpectraAutoencoder,
        np.linspace(3500, 4300, 801),
        [128, 64, 32, 16, 8],
        "autoencoder/test_8.pth",
    ),
    "test_4": AutoencoderConfig(
        SpectraAutoencoder,
        np.linspace(3500, 4300, 801),
        [16, 8, 4],
        "autoencoder/test_4.pth",
    ),}


def load_autoencoder(
    autoencoder_name: str, device: str | None = None
) -> SpectraAutoencoder:
    if autoencoder_name not in AUTOENCODER_DATABASE.keys():
        raise KeyError(f"{autoencoder_name} is not a valid model.")

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    autoencoder_config = AUTOENCODER_DATABASE[autoencoder_name]
    generator, egrid, layer_list, weights_path = (
        autoencoder_config.generator,
        autoencoder_config.egrid,
        autoencoder_config.layer_list,
        autoencoder_config.path,
    )

    autoencoder = generator(egrid, layer_list).to(device)
    autoencoder.load_state_dict(
        torch.load(join(MODEL_PATH, weights_path), weights_only=True)
    )

    return autoencoder


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    egrid = np.linspace(3500, 4300, 801)
    signal = Signal_H(egrid, geometry="S", fnoise=0)

    autoencoder = SpectraAutoencoder(egrid, layer_list=[16, 8, 4]).to(device)
    dataset_list = (
        Dataset_H_Autoencoder("Ar", signal, mode="train"),
        Dataset_H_Autoencoder("Ar", signal, mode="test"),
    )

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(autoencoder.parameters(), LEARNING_RATE)

    train_loop(
        autoencoder,
        dataset_list,
        loss_fn,
        optimizer,
        model_name=f"autoencoder_test",
    )


if __name__ == "__main__":
    main()
