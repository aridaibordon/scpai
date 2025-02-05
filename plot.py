import json

import numpy as np
import matplotlib.pyplot as plt


def plot_training_evol(path):
    with open(path, "r") as f:
        data = json.load(f)

    y = np.array([rdata["dev_mean"] for rdata in data])
    erry = np.array([rdata["dev_std"] for rdata in data])

    Te, err_Te = y[:, 0], erry[:, 0]
    rho, err_rho = y[:, 1], erry[:, 1]

    x = np.arange(len(y)) + 1

    shift = 0.075
    style = {"capsize": 3, "elinewidth": 1}

    fig, ax = plt.subplots(tight_layout=True)
    ax.errorbar(x - shift, y=Te, yerr=err_Te, fmt="o", label="Temperature", **style)
    ax.errorbar(x + shift, y=rho, yerr=err_rho, fmt="s", label="Density", **style)
    ax.axhline(0, linestyle="dashed", color="k", linewidth=1)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Relative error")
    ax.set_xticks([1] + list(range(5, 11, 5)))

    ax.legend()
    fig.savefig("test.png")


if __name__ == "__main__":
    path = "data/runs/run1.json"
    plot_training_evol(path)
