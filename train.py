import torch

import numpy as np

from scpai.config import BATCH_SIZE


def train_loop(dataloader, model, loss_fn, optimizer) -> list[float]:
    model.train()
    size = len(dataloader.dataset)

    loss_hist = []
    best = np.inf
    for batch, (x, y) in enumerate(dataloader):
        pred = model(x)
        loss = loss_fn(pred, y)
        loss_hist.append(loss.item() / BATCH_SIZE)

        if loss.item() / BATCH_SIZE < best:
            best = loss.item() / BATCH_SIZE

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item() / BATCH_SIZE, batch * BATCH_SIZE + len(x)
            print(f"\tloss: {loss:.4e}  best: {best:.4e}  [{current:>6d}/{size:>6d}]")
    
    return loss_hist


def test_loop(dataloader, model, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            pred = model(x)
            loss += loss_fn(pred, y).item() / BATCH_SIZE

    loss /= num_batches
    print(f"Test Error: Avg loss: {loss:.4e}")
