import json
import torch
import os

import numpy as np

from torch.utils.data import DataLoader

from scpai.config import BATCH_SIZE, MODEL_PATH, EPOCHS
from scpai.eval import make_dataset_prediction


def epoch_dict(epoch, loss):
    return {
        "t": epoch,
        "loss": loss,
    }


def train_loop(model, datasets, loss_fn, optimizer, model_name):
    train_dataset, test_dataset = datasets
    train_dataloader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)

    run_data = []

    best = np.inf
    for t in range(EPOCHS):
        print(f"Epoch {t + 1}")
        train_epoch(train_dataloader, model, loss_fn, optimizer)

        y, pred = make_dataset_prediction(model, test_dataset, normalized_output=True)
        loss = np.power(y - pred, 2).sum()

        if loss < best:
            best = loss
            torch.save(
                model.state_dict(), os.path.join(MODEL_PATH, f"{model_name}.pth")
            )

        # create an overview report
        run_data.append(epoch_dict(t + 1, float(loss)))
        with open(f"{model_name}.json", "w") as f:
            json.dump(run_data, f)


def train_epoch(dataloader, model, loss_fn, optimizer) -> list[float]:
    model.train()
    size = len(dataloader.dataset)

    best = np.inf
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)

        if loss.item() / BATCH_SIZE < best:
            best = loss.item() / BATCH_SIZE

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item() / BATCH_SIZE, batch * BATCH_SIZE + len(x)
            print(f"\tloss: {loss:.4e}  best: {best:.4e}  [{current:>6d}/{size:>6d}]")
