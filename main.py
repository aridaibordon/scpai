import torch

import numpy as np

import torch.nn as nn
import torch.optim as optin

from torch.utils.data import DataLoader

from model import SignalNN, SignalNNV2
from data import SignalDataset
from train import train_loop
from eval import eval_model
from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, NOISE_FACTOR


def epoch_dict(epoch, loss_hist, mean, std):
    return {
        "t": epoch,
        "loss": loss_hist,
        "dev_mean": mean.tolist(),
        "dev_std": std.tolist(),
    }


def main_loop(model, dataloader, loss_fn, optimizer, device, model_name):
    best = np.inf
    for t in range(EPOCHS):
        print(f"Epoch {t+1}")
        loss_hist = train_loop(dataloader, model, loss_fn, optimizer)

        y, pred = eval_model(device, model, t + 1)
        loss = np.abs(y - pred).sum()

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

    # select model
    model = SignalNN().to(device)

    # select dataset and dataloader
    dataset = SignalDataset("data", train=True, device=device, fnoise=NOISE_FACTOR)
    dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=True)

    loss_fn = nn.MSELoss()
    optimizer = optin.Adam(model.parameters(), LEARNING_RATE)

    # run model
    main_loop(model, dataloader, loss_fn, optimizer, device)


if __name__ == "__main__":
    main()
