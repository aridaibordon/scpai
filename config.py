import os.path

import numpy as np
import numpy.random as rd

GLOBAL_PATH = "/home/aridai/Projects/scpai"


def get_data_path(specie: str):
    return os.path.join(GLOBAL_PATH, "data/signal", specie)


MODEL_PATH = os.path.join(GLOBAL_PATH, "data/model")
RESULTS_PATH = os.path.join(GLOBAL_PATH, "data/results")


# Learning parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 1

# Signal to noise factor
NOISE_FACTOR = 0.01

# Line energy intervals (eV)
ENERGY_INTERVALS = [
    (3050, 3160),
    (3250, 3350),
    (3590, 3750),
    (3800, 4000),
    (4100, 4200),
]


def load_parameter_grid(specie: str):
    return (
        np.loadtxt(os.path.join(get_data_path(specie), "tab_tev.txt"), skiprows=1),
        np.loadtxt(os.path.join(get_data_path(specie), "tab_dne.txt"), skiprows=1),
    )


def create_data():
    dfiles = [
        fname
        for fname in os.listdir("data/data")
        if os.path.isfile(os.path.join("data/data", fname))
    ]
    dfiles = np.asarray(dfiles)

    rd.shuffle(dfiles)
    nfiles = len(dfiles)
    for ind, fname in enumerate(dfiles):
        if ind < 0.9 * nfiles:
            os.system(
                f"mv {os.path.join('data/data', fname)} {os.path.join('data/train', fname)}"
            )
        else:
            os.system(
                f"mv {os.path.join('data/data', fname)} {os.path.join('data/test', fname)}"
            )
