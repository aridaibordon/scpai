import json
import os
import torch

import numpy as np

import torch.nn as nn
import torch.optim as optin

from torch.utils.data import DataLoader

from scpai.config import MODEL_PATH, BATCH_SIZE, EPOCHS, LEARNING_RATE
from scpai.spectrum import Signal_H
from scpai.data import Dataset_H
from scpai.eval import make_dataset_prediction
from scpai.model import SCPAI_H, load_model
from scpai.train import train_loop


def epoch_dict(epoch, loss):
    return {
        "t": epoch,
        "loss": loss,
    }


def main_loop(model, datasets, loss_fn, optimizer, model_name):
    train_dataset, test_dataset = datasets
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    run_data = []

    best = np.inf
    for t in range(EPOCHS):
        print(f"Epoch {t+1}")
        train_loop(train_dataloader, model, loss_fn, optimizer)

        y, pred = make_dataset_prediction(model, test_dataset, normalized_output=True)
        loss = np.power(y - pred, 2).sum()

        # save trained weights
        # torch.save(
        #     model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}.last.pth")
        # )
        if loss < best:
            best = loss
            torch.save(
                model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}.pth")
            )

        # create an overview report
        run_data.append(epoch_dict(t + 1, loss))
        with open(f"{model_name}.json", "w") as f:
            json.dump(run_data, f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # select model
    egrid = np.linspace(3500, 4300, 800)
    signal = Signal_H(egrid, geometry="S", fnoise=0.03)

    model = SCPAI_H(egrid, layer_list=[512, 512, 512]).to(device)

    # load train and test datasets
    dataset_list = [
        Dataset_H(device, signal, train=True, assume_mass_conservation=False),
        Dataset_H(device, signal, train=False, assume_mass_conservation=False),
    ]

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(model.parameters(), LEARNING_RATE)

    # run training loop
    main_loop(model, dataset_list, loss_fn, optimizer, model_name="H_f003_n1_NMC")


if __name__ == "__main__":
    main()
