import torch
import numpy as np

from os.path import join
from torch.utils.data import DataLoader

from scpai.config import RESULTS_PATH
from scpai.data import Dataset_H
from scpai.model import MODEL_DATABASE, load_model
from scpai.spectrum import Signal_H


def make_prediction(model, signal, normalized_output: bool = False):
    print(signal)
    with torch.no_grad():
        npred = model(signal)

    ny, npred = ny.cpu().numpy(), npred.cpu().numpy()
    if normalized_output:
        return ny, npred

    y, pred = denormalize_output(ny), denormalize_output(npred)
    return y, pred


def make_dataset_prediction(model, test_dataset, normalized_output: bool = False):
    test_dataloader = DataLoader(test_dataset, 25000)
    nx, ny = next(iter(test_dataloader))
    with torch.no_grad():
        npred = model(nx)

    ny, npred = ny.cpu().numpy(), npred.cpu().numpy()
    if normalized_output:
        return ny, npred

    y, pred = denormalize_output(ny), denormalize_output(npred)
    return y, pred


def denormalize_output(output):
    nTe, nrho = output.T

    Te = 4500 * nTe + 500
    rho = 10 ** (3 * nrho + 22)
    # clength = 80e-4 * nlength + 20e-4

    return np.array([Te, rho]).T


def eval_model(model_name, fnoise, normalized_output: bool = True):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_name)

    egrid = MODEL_DATABASE[model_name]["egrid"]
    signal = Signal_H(egrid, "S", fnoise)
    test_dataset = Dataset_H(device, signal, train=False)

    y, pred = make_dataset_prediction(model, test_dataset)

    with open(join(RESULTS_PATH, model_name), "w") as f:
        np.savetxt(f, np.array([*y.T, *pred.T]).T, 2 * "%7.2f %.4e ")

    np.loadtxt(f"{join(RESULTS_PATH, model_name)}", dtype=float).T


if __name__ == "__main__":
    eval_model("H003_n1", fnoise=0)
