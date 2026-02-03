import os
import torch

import numpy as np
import numpy.random as rd

from os.path import join
from torch.utils.data import Dataset

from scpai.config import get_data_path
from scpai.spectrum import Signal_H, Signal_MZ


def get_attr_tab(specie: str):
    return (
        np.loadtxt(join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
        np.loadtxt(join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
    )


def get_file_attr(path: str, specie: str) -> tuple[int]:
    fname, _ = path.split(".")
    temp_ind, den_ind = fname.split("_")[-2:]

    data_Te = np.loadtxt(f"{get_data_path(specie)}/tab_tev.txt", skiprows=1)
    data_dne = np.loadtxt(f"{get_data_path(specie)}/tab_dne.txt", skiprows=1)

    return (
        data_Te[int(temp_ind) - 1],
        data_dne[int(den_ind) - 1],
    )


class Dataset_H(Dataset):
    def __init__(
        self,
        specie: str,
        signal: Signal_H,
        mode: str,
        assume_mass_conservation: bool = True,
        device: str | None = None,
    ) -> None:
        allowed_modes = ["train", "test"]
        if mode not in allowed_modes:
            raise ValueError(f"{mode} is not a valid mode.")

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path = join(get_data_path(specie), mode)

        self.data_Te, self.data_dne = (
            np.loadtxt(join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
            np.loadtxt(join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
        )

        self.signal = signal
        self.specie = specie
        self.assume_mass_conservation = assume_mass_conservation

    def __len__(self) -> int:
        return len(os.listdir(self.path))

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        fname = np.sort(os.listdir(self.path))[index]
        t_elec, d_elec = get_file_attr(fname, self.specie)

        if self.assume_mass_conservation:
            clength = self.estimate_characteristic_length(t_elec, d_elec)
        else:
            clength = 25e-4

        nsignal = self.signal.get_nsignal(fname=join(self.path, fname), clength=clength)
        output = self.get_normalized_output(output=[t_elec, d_elec])

        return (
            torch.Tensor(nsignal).to(self.device),
            torch.Tensor(output).to(self.device),
        )

    def estimate_characteristic_length(self, t_elec, d_elec):
        # Parameters fitted for OMEGA 2007 campaign
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
        data_dne_log10 = np.log10(self.data_dne)

        Te, dne = output
        return (
            (Te - np.min(self.data_Te)) / (np.max(self.data_Te) - np.min(self.data_Te)),
            (np.log10(dne) - np.min(data_dne_log10))
            / (np.max(data_dne_log10) - np.min(data_dne_log10)),
        )

    def get_nsignal(self, t_elec, d_elec, clength):
        mask_t = t_elec <= self.data_Te
        mask_d = d_elec <= self.data_dne

        upper_t = self.data_Te[mask_t][0]
        upper_d = self.data_dne[mask_d][0]

        t_ind = list(self.data_Te).index(upper_t)
        d_ind = list(self.data_dne).index(upper_d)

        fname = join(
            self.path, f"em_op_{self.specie}_{t_ind + 1:03d}_{d_ind + 1:03d}.txt"
        )
        return self.signal.get_nsignal(fname=join(self.path, fname), clength=clength)


class Dataset_MZ(Dataset):
    def __init__(
        self,
        specie: str,
        signal: Signal_MZ,
        clength: float,
        device: str | None = None,
    ) -> None:
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path = get_data_path(specie)

        self.data_Te, self.data_dne = (
            np.loadtxt(join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
            np.loadtxt(join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
        )

        self.signal = signal

        self.clength = clength
        self.specie = specie

    def __len__(self) -> int:
        return 50000

    def __getitem__(self, index) -> tuple[torch.Tensor]:
        t_elec_list = np.sort(rd.choice(self.data_Te, size=self.signal.nzones))[::-1]
        d_elec_list = np.sort(rd.choice(self.data_dne, size=self.signal.nzones))

        fname_list = []
        for t_elec, d_elec in zip(t_elec_list, d_elec_list):
            t_ind, d_ind = (
                np.argmax(self.data_Te == t_elec),
                np.argmax(self.data_dne == d_elec),
            )

            fname = f"em_op_{self.specie}_{t_ind + 1:03d}_{d_ind + 1:03d}.txt"
            path = (
                os.path.join(self.path, "train", fname)
                if os.path.isfile(os.path.join(self.path, "train", fname))
                else os.path.join(self.path, "test", fname)
            )

            fname_list.append(path)

        nsignal = self.signal.get_nsignal(fname_list, self.clength)
        output = np.array(
            [
                self.get_normalized_output(output=[t_elec, d_elec])
                for t_elec, d_elec in zip(t_elec_list, d_elec_list)
            ]
        )

        return (
            torch.Tensor(nsignal).to(self.device),
            torch.Tensor(output.flatten()).to(self.device),
        )

    def get_normalized_output(self, output) -> tuple[float]:
        data_dne_log10 = np.log10(self.data_dne)

        Te, dne = output
        return (
            (Te - np.min(self.data_Te)) / (np.max(self.data_Te) - np.min(self.data_Te)),
            # (dne - np.min(self.data_dne)) / (np.max(self.data_dne) - np.min(self.data_dne)),
            (np.log10(dne) - np.min(data_dne_log10))
            / (np.max(data_dne_log10) - np.min(data_dne_log10)),
        )

    def get_nsignal(
        self,
        t_elec_list,
        d_elec_list,
        clength,
        return_relative_contribution: bool = False,
    ):
        fname_list = [None for _ in range(self.signal.nzones)]
        for ind, (t_elec, d_elec) in enumerate(zip(t_elec_list, d_elec_list)):
            mask_t = t_elec <= self.data_Te
            mask_d = d_elec <= self.data_dne

            upper_t, upper_d = (
                (
                    self.data_Te[mask_t][0]
                    if self.data_Te[mask_t].any()
                    else np.max(self.data_Te)
                ),
                (
                    self.data_dne[mask_d][0]
                    if self.data_dne[mask_d].any()
                    else np.max(self.data_dne)
                ),
            )

            t_ind = list(self.data_Te).index(upper_t)
            d_ind = list(self.data_dne).index(upper_d)

            fname = f"em_op_{self.specie}_{t_ind + 1:03d}_{d_ind + 1:03d}.txt"
            fpath = (
                os.path.join(self.path, "train", fname)
                if os.path.isfile(os.path.join(self.path, "train", fname))
                else os.path.join(self.path, "test", fname)
            )

            fname_list[ind] = fpath

        if return_relative_contribution:
            return (
                self.signal.get_nsignal(fname_list, clength),
                self.signal.get_contribution_by_zone(fname_list, clength),
            )

        return self.signal.get_nsignal(fname_list, clength)
