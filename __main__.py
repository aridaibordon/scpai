import torch

import numpy as np

import torch.nn as nn
import torch.optim as optin


from scpai.config import LEARNING_RATE
from scpai.spectrum import Signal_H, Signal_MZ
from scpai.data import Dataset_H, Dataset_MZ
from scpai.model import SCPAI_H, SCPAI_MZ
from scpai.train import train_loop


def main_loop(fnoise: float):
    nzones = 5

    specie = "Ar"
    clength = 15e-4  # only used for multizone analysis

    device = "cuda" if torch.cuda.is_available() else "cpu"
    egrid = np.linspace(3200, 4000, 1601)  # Ar case
    # egrid = np.linspace(15200, 15600, 401) # Kr case

    if nzones == 1:
        signal = Signal_H(egrid, geometry="S", fnoise=fnoise)

        dataset_list = [
            Dataset_H(
                specie,
                signal,
                mode="train",
                assume_mass_conservation=False,
                device=device,
            ),
            Dataset_H(
                specie,
                signal,
                mode="train",
                assume_mass_conservation=False,
                device=device,
            ),
        ]

        model = SCPAI_H(egrid, layer_list=[512, 512, 512]).to(device)
    else:
        signal = Signal_MZ(egrid, geometry="C", nzones=nzones, fnoise=fnoise)

        dataset_list = [
            Dataset_MZ(specie, signal, clength, device=device),
            Dataset_MZ(specie, signal, clength, device=device),
        ]

        model = SCPAI_MZ(egrid, nzones=nzones, layer_list=[1024, 1024, 1024, 1024]).to(
            device
        )

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(model.parameters(), LEARNING_RATE)

    # run training loop
    train_loop(
        model,
        dataset_list,
        loss_fn,
        optimizer,
        model_name=f"Ar_MZ{nzones}_F{int(100 * fnoise):03d}_15um_nohea",
    )


def main():
    for fnoise in [0.10, 0.15, 0.20]:
        main_loop(fnoise=fnoise)


if __name__ == "__main__":
    main()
