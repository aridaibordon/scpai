import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from scpai.config import BATCH_SIZE
from spectrum import Signal_H
from scpai.data import Dataset_H
from scpai.eval import eval_model
from scpai.model import SCPAI_H, SCPAI_H2


def main() -> None:
    """
    MODEL_PATH = "data/model/omega2007_2.last.pth"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    egrid = np.linspace(3500, 4300, 800)
    signal = Signal_H(egrid, "S", fnoise=0.01)

    test_dataset = Dataset_H(device, signal, train=False)
    test_dataloader = DataLoader(test_dataset, 1, shuffle=True)

    model = SCPAI_H2(egrid).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))

    y, pred = eval_model(model, test_dataset)

    with open("data/comp_2.txt", "w") as f:
        np.savetxt(f, np.array([*y.T, *pred.T]).T, 2 * "%7.2f %.4e ")
    """

    (t_data, rho_data, t_pred, rho_pred) = np.loadtxt("data/comp.txt", dtype=float).T

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
