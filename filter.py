import numpy as np
import scipy.signal as sgn


def normal_filter(signal, delta_E: float):
    x = np.linspace(-4 * delta_E, 4 * delta_E, 12 * delta_E)
    gaussian = np.exp(-0.5 * (x / delta_E) ** 2)

    return sgn.fftconvolve(signal, gaussian, mode="same")
