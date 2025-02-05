import torch
import numpy as np

from data import load_data


def eval_model(device, model, epoch: int):
    _, test_dataloader = load_data(device, batch_size=10000)

    nx, ny = next(iter(test_dataloader))
    with torch.no_grad():
        npred = model(nx)

    ny, npred = ny.cpu().numpy(), npred.cpu().numpy()
    y, pred = denormalize_output(ny), denormalize_output(npred)

    return y, pred


def denormalize_output(out: np.array):
    nTe_data, nrho_data, ntau_data = out[:, 0], out[:, 1], out[:, 2]

    Te_data = 4500 * nTe_data + 500
    rho_data = 10 ** (3 * nrho_data + 22)
    tau_data = 80e-4 * ntau_data + 20e-4

    return np.asarray(
        [np.array([Te, rho, tau]) for Te, rho, tau in zip(Te_data, rho_data, tau_data)]
    )
