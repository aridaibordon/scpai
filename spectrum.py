import numpy as np
import numpy.random as rd
import scipy.signal as sgn


INS_RESOLUTION = 10  # eV


def load_radiative_properties_file(fname: str):
    raw_data = np.loadtxt(fname).T

    keys = ["E", "bb", "bf", "ff", "emi", "obb", "obf", "off", "opa"]
    data = {}
    for key, row in zip(keys, raw_data[:-2]):
        data[key] = row

    return data["E"], data["bb"] + data["bf"], data["opa"]


def apply_instrument_resolution(egrid, signal, delta_E: float):
    x = np.arange(-4 * delta_E, 4 * delta_E, egrid[1] - egrid[0])
    gaussian = np.exp(-0.5 * (x / delta_E) ** 2)

    return sgn.fftconvolve(signal, gaussian, mode="same")


class Signal_H:
    def __init__(self, egrid: list, geometry: str, fnoise: float):
        self.egrid = egrid
        self.fnoise = fnoise

        valid_geometries = ["P", "S"]
        if geometry not in valid_geometries:
            raise Exception(f"Geometry symbol {geometry} not recognized.")

        self.geometry = geometry

    def get_nsignal(self, fname: str, clength: float):
        egrid, j, k = load_radiative_properties_file(fname)

        j = np.interp(self.egrid, egrid, j)
        k = np.interp(self.egrid, egrid, k)

        if self.geometry == "P":
            signal = (j / k) * (1 - np.exp(-k * clength))
        elif self.geometry == "S":
            signal = (
                np.pi
                * clength**2
                * (j / k)
                * (
                    1
                    + np.exp(-2 * k * clength) / (k * clength)
                    - (1 - np.exp(-2 * k * clength)) / (2 * (k * clength) ** 2)
                )
            )

        # noise = self.fnoise * max(signal) * (2 * rd.random(len(signal)) - 1)
        noise = rd.normal(loc=0, scale=self.fnoise * max(signal))
        signal += noise

        signal = apply_instrument_resolution(egrid, signal, INS_RESOLUTION / 2.355)
        return (signal - min(signal)) / (max(signal) - min(signal))


class Signal_MZ:
    def __init__(self, egrid: list, geometry: str, nzones: int, fnoise: float):
        self.egrid = egrid

        self.nzones = nzones
        self.fnoise = fnoise

        valid_geometries = ["P"]
        if geometry not in valid_geometries:
            raise Exception(f"Geometry symbol {geometry} not recognized.")

        self.geometry = geometry

    def get_nsignal(self, fname_list: list, clength: float):
        nzones = len(fname_list)

        egrid, j, k = np.array(
            [load_radiative_properties_file(fname) for fname in fname_list]
        ).T

        signal = np.zeros(len(egrid))
        for ind in range(nzones):
            j[ind] / k[ind] * (1 - np.exp(-k[ind] * clength))
