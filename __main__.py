import torch

import numpy as np

import torch.nn as nn
import torch.optim as optin


from scpai.config import LEARNING_RATE
from scpai.spectrum import Signal_H
from scpai.data import Dataset_H
from scpai.model import SCPAI_H
from scpai.train import train_loop


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    egrid = np.linspace(3500, 4300, 800)
    signal = Signal_H(egrid, geometry="S", fnoise=0.03)

    # load train and test datasets
    dataset_list = [
        Dataset_H(signal, mode="train", assume_mass_conservation=True, device=device),
        Dataset_H(signal, mode="test", assume_mass_conservation=True, device=device),
    ]

    # select model
    model = SCPAI_H(egrid, layer_list=[512, 512, 512, 512, 512]).to(device)

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(model.parameters(), LEARNING_RATE)

    # run training loop
    train_loop(model, dataset_list, loss_fn, optimizer, model_name="H_f003_n2_MC")


if __name__ == "__main__":
    main()
