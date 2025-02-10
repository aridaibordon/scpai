import torch

import numpy as np

import torch.nn as nn
import torch.optim as optin

from torch.utils.data import DataLoader

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE
from csignal import Signal_H
from data import Dataset_H
from eval import eval_model
from model import SCPAI_H
from train import train_loop


def epoch_dict(epoch, loss_hist, mean, std):
    return {
        "t": epoch,
        "loss": loss_hist,
        "dev_mean": mean.tolist(),
        "dev_std": std.tolist(),
    }


def main_loop(model, datasets, loss_fn, optimizer, device, model_name):
    train_dataset, test_dataset = datasets
    train_dataloader, test_dataloader = (
        DataLoader(train_dataset, BATCH_SIZE, shuffle=True),
        DataLoader(test_dataset, BATCH_SIZE, shuffle=True),
    )

    best = np.inf
    for t in range(EPOCHS):
        print(f"Epoch {t+1}")
        loss_hist = train_loop(train_dataloader, model, loss_fn, optimizer)

        y, pred = eval_model(test_dataloader, model)
        loss = np.power(y - pred, 2).sum()

        # save trained weights
        torch.save(model.state_dict(), f"model/{model_name}.last.pth")
        if loss < best:
            best = loss
            torch.save(model.state_dict(), f"model/{model_name}.best.pth")

        # create an overview report
        # run_data.append(epoch_dict(t + 1, loss_hist, loss))
        # with open(f"{run_name}.json", "w") as f:
        #     json.dump(run_data, f)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    egrid = np.linspace(3000, 4500, 3000)
    signal = Signal_H(egrid, "S", fnoise=0.01)

    # select model
    model = SCPAI_H(egrid).to(device)

    # load train and test datasets
    dataset_list = [
        Dataset_H(device, signal, train=True),
        Dataset_H(device, signal, train=False),
    ]

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(model.parameters(), LEARNING_RATE)

    # run model
    main_loop(model, dataset_list, loss_fn, optimizer, device, "test")


if __name__ == "__main__":
    main()
