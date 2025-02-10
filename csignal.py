import numpy as np
import numpy.random as rd

from abc import ABC

from filter import apply_instrument_resolution


INS_RESOLUTION = 10  # eV


class BaseSignal(ABC):
    def __init__(self, egrid: list, geometry: str, fnoise: float):
        self.egrid = egrid
        self.fnoise = fnoise

        valid_geometries = ["P", "S"]
        if geometry not in valid_geometries:
            pass

        self.geometry = geometry

    def get_nsignal(self, rad_prop: dict, clength: float): ...


class Signal_H:
    def __init__(
        self,
        egrid: list,
        geometry: str,
        t_elec: float,
        d_elec: float,
        clength: float,
        fnoise: float,
    ):
        self.t_elec, self.d_elec, self.clength = t_elec, d_elec, clength

        self.egrid = egrid
        self.fnoise = fnoise

        valid_geometries = ["P", "S"]
        if geometry not in valid_geometries:
            pass

        self.geometry = geometry

    def load_radiative_properties(self, fname: str):
        raw_data = np.loadtxt(fname).T
        keys = ["E", "bb", "bf", "ff", "emi", "obb", "obf", "off", "opa"]

        data = {}
        for key, row in zip(keys, raw_data[:-2]):
            data[key] = row

        return data["E"], data["bb"], data["opa"]

    def get_nsignal(self, fname: str, clength: float):
        egrid, j_bb, k = self.load_radiative_properties(fname)

        j_bb = np.interp(self.egrid, egrid, j_bb)
        k = np.interp(self.egrid, egrid, k)

        # compute signal
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

        # add noise
        signal = signal / max(signal)
        if self.fnoise:
            noise = self.fnoise * (2 * rd.random(len(signal)) - 1)
            signal += noise

        # postprocessing and normalization
        signal = apply_instrument_resolution(egrid, signal, INS_RESOLUTION / 2.355)
        return (signal - min(signal)) / (max(signal) - min(signal))
