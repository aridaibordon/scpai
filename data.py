import os
import torch

import numpy as np

from os.path import join
from torch.utils.data import Dataset

from csignal import Signal_H

DATA_PATH = "data/"


def get_file_attr(path: str) -> tuple[int]:
    fname, _ = path.split(".")
    temp_ind, den_ind = fname.split("_")[-2:]

    data_Te = np.loadtxt(f"{DATA_PATH}/tab_tev.txt", skiprows=1)
    data_dne = np.loadtxt(f"{DATA_PATH}/tab_dne.txt", skiprows=1)

    return (
        data_Te[int(temp_ind) - 1],
        data_dne[int(den_ind) - 1],
    )


class Dataset_H(Dataset):
    def __init__(self, device: str, signal: Signal_H, train: bool) -> None:
        self.device = device

        self.path = join(DATA_PATH, "train" if train else "test")

        self.data_Te = np.loadtxt(join(DATA_PATH, "tab_tev.txt"), skiprows=1)
        self.data_dne = np.loadtxt(join(DATA_PATH, "tab_dne.txt"), skiprows=1)

        self.signal = signal

    def __len__(self) -> int:
        return len(os.listdir(self.path))

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        fname = np.sort(os.listdir(self.path))[index]
        t_elec, d_elec = get_file_attr(fname)

        nlength = 0.5  #  rd.random()
        clength = self.denormalize_clenght(nlength)

        nsignal = self.signal.get_nsignal(fname=join(self.path, fname), clength=clength)

        ytrain = np.array([t_elec, d_elec, nlength])
        output = self.get_normalized_output(ytrain)

        return (
            torch.Tensor(nsignal).to(self.device),
            torch.Tensor(output).to(self.device),
        )

    def denormalize_clenght(self, nlength):
        min_val, max_val = 20e-4, 100e-4
        return (max_val - min_val) * nlength + min_val

    def get_normalized_output(self, output) -> tuple[float]:
        Te, dne, nlength = output
        return (
            (Te - 500) / (5000 - 500),
            (np.log10(dne) - 22) / (25 - 22),
            nlength,
        )


def get_datasets_h(device, signal):
    return (
        Dataset_H(device, signal, train=True),
        Dataset_H(device, signal, train=False),
    )
