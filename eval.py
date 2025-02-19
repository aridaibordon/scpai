import torch
import numpy as np

from torch.utils.data import DataLoader


def eval_model(model, test_dataset):
    test_dataloader = DataLoader(test_dataset, 25000)
    nx, ny = next(iter(test_dataloader))
    with torch.no_grad():
        npred = model(nx)

    ny, npred = ny.cpu().numpy(), npred.cpu().numpy()
    y, pred = denormalize_output(ny), denormalize_output(npred)

    return y, pred


def denormalize_output(output):
    nTe, nrho, nlength = output.T

    Te = 4500 * nTe + 500
    rho = 10 ** (3 * nrho + 22)
    clength = 80e-4 * nlength + 20e-4

    return np.array([Te, rho, clength]).T
