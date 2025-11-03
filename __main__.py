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
    uniform_case = True
    specie = "Ar"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    egrid = np.linspace(3500, 4300, 800)

    if uniform_case:
        signal = Signal_H(egrid, geometry="S", fnoise=fnoise)

        dataset_list = [
            Dataset_H(
                specie, signal, mode="train", assume_mass_conservation=True, device=device
            ),
            Dataset_H(
                specie, signal, mode="test", assume_mass_conservation=True, device=device
            ),
        ]

        model = SCPAI_H(egrid, layer_list=[512, 512, 512]).to(device)
    else:
        signal = Signal_MZ(egrid, geometry="C", nzones=3, fnoise=fnoise)

        dataset_list = [
            Dataset_MZ(
                specie, signal, mode="train", assume_mass_conservation=True, device=device
            ),
            Dataset_MZ(
                specie, signal, mode="test", assume_mass_conservation=True, device=device
            ),
        ]

        model = SCPAI_MZ(egrid, nzones=3, layer_list=[512, 512, 512, 512, 512]).to(
            device
        )

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(model.parameters(), LEARNING_RATE)

    # run training loop
    train_loop(model, dataset_list, loss_fn, optimizer, model_name=f"Ar_H_f{fnoise}_MC")


def main():
    for fnoise in [0.05]:
        main_loop(fnoise=fnoise)


if __name__ == "__main__":
    main()
