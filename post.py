import numpy as np
import scipy.signal as sgn


def apply_instrument_resolution(egrid, signal, delta_E: float):
    x = np.arange(-4 * delta_E, 4 * delta_E, egrid[1] - egrid[0])
    gaussian = np.exp(-0.5 * (x / delta_E) ** 2)

    return sgn.fftconvolve(signal, gaussian, mode="same")