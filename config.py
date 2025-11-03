import os

import numpy as np
import numpy.random as rd

from os.path import isfile, join


GLOBAL_PATH = "/home/aridai/PhD/research/projects/scpai"

def get_data_path(specie: str):
    return join(GLOBAL_PATH, "data/signal", specie)

MODEL_PATH = join(GLOBAL_PATH, "data/model")
RESULTS_PATH = join(GLOBAL_PATH, "data/results")


# Learning parameters
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 10

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


def create_data():
    dfiles = [
        fname for fname in os.listdir("data/data") if isfile(join("data/data", fname))
    ]
    dfiles = np.asarray(dfiles)

    rd.shuffle(dfiles)
    nfiles = len(dfiles)
    for ind, fname in enumerate(dfiles):
        if ind < 0.9 * nfiles:
            os.system(f"mv {join('data/data', fname)} {join('data/train', fname)}")
        else:
            os.system(f"mv {join('data/data', fname)} {join('data/test', fname)}")
