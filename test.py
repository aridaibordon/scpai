import torch
import matplotlib.pyplot as plt
import numpy as np

from os.path import join

from scpai.config import RESULTS_PATH
from scpai.data import Dataset_H
from scpai.eval import make_dataset_prediction
from scpai.model import load_model
from scpai.spectrum import Signal_H


MODEL_NAME = "H003_n1"


def main() -> None:
    # egrid = np.linspace(3500, 4300, 800)
    # signal = Signal_H(egrid, "S", fnoise=0.01)

    # test_dataset = Dataset_H(signal, mode="test")

    # model = load_model(MODEL_NAME)

    # y, pred = make_dataset_prediction(model, test_dataset)

    # with open("data/comp_2.txt", "w") as f:
    #     np.savetxt(f, np.array([*y.T, *pred.T]).T, 2 * "%7.2f %.4e ")

    (t_data, rho_data, t_pred, rho_pred) = np.loadtxt(
        f"{join(RESULTS_PATH, MODEL_NAME)}.txt", dtype=float
    ).T

    mask = t_data < 3000

    t_data, rho_data = t_data[mask] / 1e3, rho_data[mask]
    t_pred, rho_pred = t_pred[mask] / 1e3, rho_pred[mask]

    print(np.std(t_data - t_pred))
    print(np.std(np.log10(rho_data) - np.log10(rho_pred)))

    # temperature comparison
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.plot(t_data, t_pred, ".", ms=1.5)
    ax.plot((min(t_data), max(t_data)), (min(t_data), max(t_data)), "k-", lw=3)

    ax.set_xlabel("True temperature (keV)")
    ax.set_ylabel("Predicted temperature (keV)")

    r2 = 1 - np.sum((t_pred - t_data) ** 2) / np.sum((t_pred - np.mean(t_pred)) ** 2)
    print(r2)

    fig.savefig("comp_tev.png", dpi=300)

    # density comparison
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.plot(rho_data, rho_pred, ".", ms=1.5)
    ax.plot((min(rho_data), max(rho_data)), (min(rho_data), max(rho_data)), "k-", lw=3)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("True electron density (cm$^{-3}$)")
    ax.set_ylabel("Predicted electron density (cm$^{-3}$)")

    r2 = 1 - np.sum((rho_pred - rho_data) ** 2) / np.sum(
        (rho_pred - np.mean(rho_pred)) ** 2
    )
    print(r2)

    fig.savefig("comp_dne.png", dpi=300)

    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)

    sc = ax.scatter(
        t_data, rho_data, c=np.abs(t_data - t_pred) * 1e3, s=4, vmin=0, vmax=200
    )
    fig.colorbar(sc)

    ax.set_xlim(0.5, 3.0)
    ax.set_ylim(1e22, 1e25)

    ax.set_yscale("log")

    fig.savefig("test.png")

    # characteristic length comparison
    # fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    # ax.plot(tau_data, tau_pred, ".", ms=1)
    # ax.plot((min(tau_data), max(tau_data)), (min(tau_data), max(tau_data)), "k-")

    # ax.set_xlabel("Characteristic plasma length")
    # ax.set_ylabel("Predicted characteristic plasma length")

    # fig.savefig("comp_tau.png")


if __name__ == "__main__":
    main()
