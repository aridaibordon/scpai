import numpy as np
import numpy.random as rd

from filter import apply_instrument_resolution


INS_RESOLUTION = 10  # eV


def load_radiative_properties_file(fname: str):
    raw_data = np.loadtxt(fname).T

    keys = ["E", "bb", "bf", "ff", "emi", "obb", "obf", "off", "opa"]
    data = {}
    for key, row in zip(keys, raw_data[:-2]):
        data[key] = row

    return data["E"], data["bb"], data["opa"]


class Signal_H:
    def __init__(self, egrid: list, geometry: str, fnoise: float):
        self.egrid = egrid
        self.fnoise = fnoise

        valid_geometries = ["P", "S"]
        if geometry not in valid_geometries:
            raise Exception(f"Geometry symbol {geometry} not recognized.")

        self.geometry = geometry

    def get_nsignal(self, fname: str, clength: float):
        egrid, j_bb, k = load_radiative_properties_file(fname)

        j_bb = np.interp(self.egrid, egrid, j_bb)
        k = np.interp(self.egrid, egrid, k)

        if self.geometry == "P":
            signal = (j_bb / k) * (1 - np.exp(-k * clength))
        elif self.geometry == "S":
            signal = (
                np.pi
                * clength**2
                * (j_bb / k)
                * (
                    1
                    + np.exp(-2 * k * clength) / (k * clength)
                    - (1 - np.exp(-2 * k * clength)) / (2 * (k * clength) ** 2)
                )
            )

        signal = signal / max(signal)
        if self.fnoise:
            noise = self.fnoise * (2 * rd.random(len(signal)) - 1)
            signal += noise

        signal = apply_instrument_resolution(egrid, signal, INS_RESOLUTION / 2.355)
        return (signal - min(signal)) / (max(signal) - min(signal))
