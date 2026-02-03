import torch
import numpy as np

from os.path import join
from torch.utils.data import DataLoader

from scpai.config import RESULTS_PATH
from scpai.data import Dataset_H
from scpai.model import MODEL_H_DATABASE, MODEL_MZ_DATABASE, load_h_model, load_mz_model
from scpai.spectrum import Signal_H


def predict_h(x_exp, y_exp, model_name):
    egrid = MODEL_H_DATABASE[model_name].egrid

    model = load_h_model(model_name, device="cpu")

    signal = np.interp(egrid, x_exp, y_exp)

    y_norm = torch.from_numpy(
        (signal - min(signal)) / (max(signal) - min(signal))
    ).float()

    with torch.no_grad():
        n_pred = model(y_norm)

    return denormalize_output(n_pred)


def predict_mz(x_exp, y_exp, model_name):
    egrid = MODEL_MZ_DATABASE[model_name].egrid

    model = load_mz_model(model_name, device="cpu")

    signal = np.interp(egrid, x_exp, y_exp)

    y_norm = torch.from_numpy(
        (signal - min(signal)) / (max(signal) - min(signal))
    ).float()

    with torch.no_grad():
        n_pred = model(y_norm)

    return [denormalize_output(out) for out in n_pred.reshape(-1, 2)]


def make_dataset_prediction(model, test_dataset, normalized_output: bool = False):
    test_dataloader = DataLoader(test_dataset, 50000)
    nx, ny = next(iter(test_dataloader))

    model.eval()
    with torch.no_grad():
        npred = model(nx)

    ny, npred = ny.cpu().numpy(), npred.cpu().numpy()
    if normalized_output:
        return ny, npred

    y, pred = (
        denormalize_output(test_dataset.specie, ny),
        denormalize_output(test_dataset.specie, npred),
    )
    return y, pred


def denormalize_output(output):
    nTe, nrho = output

    Te = 6600 * nTe + 400
    rho = 10 ** (1.69897 * nrho - 1)
    # rho = 4.9 * nrho + 0.1

    return np.array([Te, rho]).T


def eval_model(model_name, fnoise):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_h_model(model_name)

    egrid = MODEL_H_DATABASE[model_name]["egrid"]
    signal = Signal_H(egrid, "S", fnoise)
    test_dataset = Dataset_H(device, signal, mode=False)

    y, pred = make_dataset_prediction(model, test_dataset)

    with open(join(RESULTS_PATH, model_name), "w") as f:
        np.savetxt(f, np.array([*y.T, *pred.T]).T, 2 * "%7.2f %.4e ")

    # return np.loadtxt(f"{join(RESULTS_PATH, model_name)}", dtype=float).T


if __name__ == "__main__":
    eval_model("paper_Ar_H_f000", fnoise=0)
