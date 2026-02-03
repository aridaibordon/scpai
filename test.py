import torch
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

from os.path import join

from scpai.config import RESULTS_PATH
from scpai.data import Dataset_H, Dataset_MZ
from scpai.eval import make_dataset_prediction
from scpai.model import load_h_model, load_mz_model, load_mz_config
from scpai.spectrum import Signal_H, Signal_MZ
from scpai.utils import denormalize_output


MODEL_NAME = "test_Kr_MZ2_010"


def test_MZ(specie: str, model_name: str) -> None:
    # clength has to be manually setted
    # nzones has to be manually setted

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = load_mz_config(model_name)

    egrid = model_config.egrid
    signal = Signal_MZ(egrid, "S", nzones=2, fnoise=0)

    # clength must be manually updated
    dataset = Dataset_MZ(specie=specie, clength=25e-4, signal=signal, device=device)
    model = load_mz_model(model_name)

    y, pred = make_dataset_prediction(model, dataset, normalized_output=True)

    with open("data/comp.txt", "w") as f:
        np.savetxt(f, np.array([*y.T, *pred.T]).T, "%.4e")


def main() -> None:
    egrid = np.linspace(3500, 4300, 800)
    signal = Signal_H(egrid, "S", fnoise=0)

    test_dataset = Dataset_H("Ar", signal)

    model = load_h_model(MODEL_NAME)

    y, pred = make_dataset_prediction(model, test_dataset)

    with open("data/comp_2.txt", "w") as f:
        np.savetxt(f, np.array([*y.T, *pred.T]).T, 2 * "%7.2f %.4e ")

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
    model_config = load_mz_config(MODEL_NAME)

    specie = "Kr"
    egrid = model_config.egrid

    # test_MZ(specie, model_name=MODEL_NAME)

    data = np.loadtxt("data/comp.txt")

    nzones = data.shape[1] // 4
    clength = 20e-4

    signal = Signal_MZ(egrid, "S", nzones, fnoise=0)
    dataset = Dataset_MZ(specie, signal, clength)

    cond = np.zeros((2 * nzones * len(data), 2), dtype=object)
    for ind, ncond in enumerate(data.flatten().reshape(-1, 2)):
        cond[ind] = denormalize_output(ncond)

    cond = cond.reshape((len(data), 2, nzones, -1))

    for zone in range(nzones):
        print(zone)
        zone_t_elec = cond[:, 0, zone, 0] / 1e3
        zone_d_elec = cond[:, 0, zone, 1]

        zone_t_elec_pred = cond[:, 1, zone, 0] / 1e3
        zone_d_elec_pred = cond[:, 1, zone, 1]

        for ind in range(100):
            t_elec_list = cond[ind, 0, :, 0]
            t_elec_list_pred = cond[ind, 1, :, 0]

            d_elec_list = cond[ind, 0, :, 1]
            d_elec_list_pred = cond[ind, 1, :, 1]

            signal_pred = dataset.get_nsignal(
                t_elec_list_pred,
                d_elec_list_pred,
                clength,
            )
            signal_real = dataset.get_nsignal(
                t_elec_list,
                d_elec_list,
                clength,
            )

            zone_pos = [
                i * clength / nzones * 1e4 for i in range(nzones + 1) for _ in range(2)
            ][1:-1]

            t_elec_zone = [t_elec / 1e3 for t_elec in t_elec_list for _ in range(2)]
            t_elec_zone_pred = [
                t_elec / 1e3 for t_elec in t_elec_list_pred for _ in range(2)
            ]

            d_elec_zone = [d_elec for d_elec in d_elec_list for _ in range(2)]
            d_elec_zone_pred = [d_elec for d_elec in d_elec_list_pred for _ in range(2)]

            fig = plt.figure(figsize=(5, 5), tight_layout=True)
            gs = gridspec.GridSpec(2, 2)

            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])

            ax1.plot(egrid / 1e3, signal_pred, label="pred")
            ax1.plot(egrid / 1e3, signal_real, "--", label="test")

            # ax1.set_xlim(3550, 4250)

            ax1.legend()

            ax1.set_xlabel("Photon energy (eV)")
            ax1.set_ylabel("Intensity (arb. units)")

            ax2.plot(zone_pos, t_elec_zone_pred)
            ax2.plot(zone_pos, t_elec_zone, "--")

            ax2.set_ylim(0.5, 5.0)

            ax2.set_xlabel("Core radius ($\mu$m)")
            ax2.set_ylabel("Electron temperature (keV)")

            ax3.plot(zone_pos, d_elec_zone_pred)
            ax3.plot(zone_pos, d_elec_zone, "--")

            ax3.set_ylim(1e22, 1e25)

            ax3.set_yscale("log")

            ax3.set_xlabel("Core radius ($\mu$m)")
            ax3.set_ylabel("Electron density (cm$^{-3}$)")

            fig.savefig(f"plot/mz_Kr/test{ind}.png")
            plt.close(fig)
