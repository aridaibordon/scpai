import os

import numpy as np

from scpai.config import get_data_path
from scpai.data import get_file_attr
from scpai.spectrum import Signal_H


def denormalize_output(output):
    nTe, nrho = output.T

    Te = 4500 * nTe + 500
    rho = 10 ** (3 * nrho + 22)

    return np.array([Te, rho]).T


def estimate_characteristic_length(t_elec, d_elec):
    # Parameters fitted for OMEGA 2007 campaign
    p_ar, p_d2 = 0.072, 20

    ro = 400e-4
    to = 300
    kb = 1.3806503e-23

    p_d2_si = 0.072 * 101325
    d_elec_si = d_elec * 1e6
    zbar = 16  # approximated zbar in conditions range

    return (((2 + zbar * p_ar / p_d2) * p_d2_si) / (kb * to * d_elec_si)) ** (
        1 / 3
    ) * ro


class SignalDataset:
    def __init__(self, specie: str, signal: Signal_H):
        self.specie = specie
        self.signal = signal

        self.data_Te, self.data_dne = (
            np.loadtxt(os.path.join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
            np.loadtxt(os.path.join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
        )

        self.t_size, self.d_size = len(self.t_elec_list), len(self.d_elec_list)

    def __len__(self):
        return self.t_size * self.d_size

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"index {index} is out of bound (max {len(self)}).")

        t_ind = index // self.d_size + 1
        d_ind = index % self.d_size + 1

        fname = f"em_op_Ar_{t_ind:03d}_{d_ind:03d}.txt"
        t_elec, d_elec = get_file_attr(fname)
        clength = estimate_characteristic_length(t_elec, d_elec)

        data_path = get_data_path(self.specie)

        path = (
            os.path.join(data_path, "train", fname)
            if os.path.isfile(os.path.join(data_path, "train", fname))
            else os.path.join(data_path, "test", fname)
        )

        return (
            self.signal.egrid,
            self.signal.get_nsignal(path, clength),
            (t_elec, d_elec),
        )

    def get_signal(self, t_elec, d_elec):
        mask_t = t_elec < self.t_elec_list
        mask_d = d_elec < self.d_elec_list

        near_t = self.t_elec_list[mask_t][0]
        near_d = self.d_elec_list[mask_d][0]

        t_ind = list(self.t_elec_list).index(near_t)
        d_ind = list(self.d_elec_list).index(near_d)

        return self[d_ind + t_ind * self.t_size]
