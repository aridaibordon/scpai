import torch
import matplotlib.pyplot as plt
import numpy as np

from os.path import join

from scpai.data import Dataset_H
from scpai.eval import eval_model
from scpai.model import load_model
from scpai.spectrum import Signal_H


RESULTS_PATH = "data/results"
MODEL_NAME = "H003_n1"


def main() -> None:
    (t_data, rho_data, t_pred, rho_pred) = np.loadtxt(
        f"{join(RESULTS_PATH, MODEL_NAME)}.txt", dtype=float
    ).T

    mask = t_data < 3000

    t_data, rho_data = t_data[mask], rho_data[mask]
    t_pred, rho_pred = t_pred[mask], rho_pred[mask]

    print(np.std(t_data - t_pred))
    print(np.std(np.log10(rho_data) - np.log10(rho_pred)))

    # temperature comparison
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.plot(t_data, t_pred, ".", ms=2)
    ax.plot((min(t_data), max(t_data)), (min(t_data), max(t_data)), "k-")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Predicted temperature")

    fig.savefig("comp_tev.png")

    # density comparison
    fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    ax.plot(rho_data, rho_pred, ".", ms=1)
    ax.plot((min(rho_data), max(rho_data)), (min(rho_data), max(rho_data)), "k-")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel("Density")
    ax.set_ylabel("Predicted density")

    fig.savefig("comp_dne.png")

    # characteristic length comparison
    # fig, ax = plt.subplots(figsize=(5, 5), tight_layout=True)
    # ax.plot(tau_data, tau_pred, ".", ms=1)
    # ax.plot((min(tau_data), max(tau_data)), (min(tau_data), max(tau_data)), "k-")

    # ax.set_xlabel("Characteristic plasma length")
    # ax.set_ylabel("Predicted characteristic plasma length")

    # fig.savefig("comp_tau.png")


if __name__ == "__main__":
    main()
