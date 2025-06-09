import os
import torch

import numpy as np
import numpy.random as rd

from os.path import join
from torch.utils.data import Dataset

from scpai.config import DATA_PATH
from scpai.spectrum import Signal_H, Signal_MZ


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
    def __init__(
        self,
        device: str,
        signal: Signal_H,
        train: bool,
        assume_mass_conservation: bool = True,
    ) -> None:
        self.device = device

        self.path = join(DATA_PATH, "train" if train else "test")

        self.data_Te = np.loadtxt(join(DATA_PATH, "tab_tev.txt"), skiprows=1)
        self.data_dne = np.loadtxt(join(DATA_PATH, "tab_dne.txt"), skiprows=1)

        self.signal = signal

        self.assume_mass_conservation = assume_mass_conservation

    def __len__(self) -> int:
        return len(os.listdir(self.path))

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        fname = np.sort(os.listdir(self.path))[index]
        t_elec, d_elec = get_file_attr(fname)

        if self.assume_mass_conservation:
            clength = self.estimate_characteristic_length(t_elec, d_elec)
        else:
            clength = 80e-4 * rd.uniform() + 20e-4

        nsignal = self.signal.get_nsignal(fname=join(self.path, fname), clength=clength)

        ytrain = np.array([t_elec, d_elec])
        output = self.get_normalized_output(ytrain)

        return (
            torch.Tensor(nsignal).to(self.device),
            torch.Tensor(output).to(self.device),
        )

    # Parameters fitted for OMEGA 2007 campaign
    def estimate_characteristic_length(self, t_elec, d_elec):
        p_ar, p_d2 = 0.072, 20

        ro = 400e-4
        to = 300
        kb = 1.3806503e-23

        p_d2_si = p_d2 * 101325
        d_elec_si = d_elec * 1e6
        zbar = 16  # approximated zbar in conditions range

        return (((2 + zbar * p_ar / p_d2) * p_d2_si) / (kb * to * d_elec_si)) ** (
            1 / 3
        ) * ro

    def denormalize_clenght(self, nlength):
        min_val, max_val = 20e-4, 100e-4
        return (max_val - min_val) * nlength + min_val

    def get_normalized_output(self, output) -> tuple[float]:
        Te, dne = output
        return (
            (Te - 500) / (5000 - 500),
            (np.log10(dne) - 22) / (25 - 22),
        )


class Dataset_MZ(Dataset):
    def __init__(self, device: str, signal: Signal_MZ, train: bool) -> None:
        self.device = device

        self.path = join(DATA_PATH, "train" if train else "test")

        self.data_Te = np.loadtxt(join(DATA_PATH, "tab_tev.txt"), skiprows=1)
        self.data_dne = np.loadtxt(join(DATA_PATH, "tab_dne.txt"), skiprows=1)

        self.signal = signal

    def __len__(self) -> int:
        return 100000

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        sel_Te = np.random.choice(len(self.data_Te))
        sel_dne = np.random.choice(len(self.data_Te))
