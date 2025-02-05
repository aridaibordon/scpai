import os
import torch

import pandas as pd
import numpy as np
import numpy.random as rd

from torch.utils.data import Dataset, DataLoader

from filter import normal_filter
from config import BATCH_SIZE, ENERGY_INTERVALS


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


class SignalDataset(Dataset):
    def __init__(
        self, path: str, train: bool, device: str, fnoise: float | None = None
    ) -> None:
        self.device = device

        self.path = f"{path}/train" if train else f"{path}/test"
        self.data_Te = np.loadtxt(f"{path}/tab_tev.txt", skiprows=1)
        self.data_dne = np.loadtxt(f"{path}/tab_dne.txt", skiprows=1)

        self.fnoise = fnoise

    def __len__(self) -> int:
        return len(os.listdir(self.path))

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        fname = np.sort(os.listdir(self.path))[index]
        t_elec, d_elec = get_file_attr(fname)

        nlength = rd.random()
        clength = self.normalize_clenght(nlength)

        df = self.load_data_from_file(fname)
        # df = self.select_lines(df)

        signal = self.get_signal(df, clength)
        signal = self.add_noise(signal) if self.fnoise else signal
        signal = self.apply_filter(signal)
        signal = self.normalize_signal(signal)

        ytrain = np.array([t_elec, d_elec, nlength])
        output = self.get_normalized_output(ytrain)

        return (
            torch.Tensor(signal).to(self.device),
            torch.Tensor(output).to(self.device),
        )

    def normalize_clenght(self, clength):
        """Get physical length for normalized length"""
        # return 20e-4
        min_val, max_val = 20e-4, 100e-4
        return (max_val - min_val) * clength + min_val

    def load_data_from_file(self, fname: str) -> pd.DataFrame:
        fpath = f"{self.path}/{fname}"

        with open(fpath, "r") as f:
            data = [line.split()[:-2] for line in f]

        keys = ["E", "bb", "bf", "ff", "emi", "obb", "obf", "off", "opa"]
        return pd.DataFrame(data, columns=keys, dtype=np.float32)

    def select_lines(self, df: pd.DataFrame) -> pd.DataFrame:
        selection = []
        for start, end in ENERGY_INTERVALS:
            mask = df["E"].between(start, end)
            selection.append(df[mask])

        return pd.concat(selection)

    def get_signal(self, df: pd.DataFrame, clength: float = 50e-4) -> np.array:
        df["opa"] = df["obb"] + df["obf"] + df["off"]
        signal = (df["bb"] / (df["opa"])) * (1 - np.exp(-df["opa"] * clength))

        return signal.to_numpy() / max(signal)

    def add_noise(self, signal) -> np.array:
        noise = self.fnoise * (2 * rd.random(len(signal)) - 1)
        return signal + noise

    def apply_filter(self, signal) -> np.array:
        return normal_filter(signal, delta_E=10)

    def normalize_signal(self, signal) -> np.array:
        return (signal - min(signal)) / (max(signal) - min(signal))

    def get_normalized_output(self, output) -> tuple[float]:
        Te, dne, length = output
        return (
            (Te - 500) / (5000 - 500),
            (np.log10(dne) - 22) / (25 - 22),
            length,
        )


def load_data(device: str, batch_size: int | None = None) -> tuple[DataLoader]:
    """Return test and train dataloader"""
    if batch_size == None:
        batch_size = BATCH_SIZE

    train_dataset = SignalDataset("data", train=True, device=device)
    test_dataset = SignalDataset("data", train=False, device=device)

    return (
        DataLoader(train_dataset, batch_size, shuffle=True),
        DataLoader(test_dataset, batch_size, shuffle=True),
    )
