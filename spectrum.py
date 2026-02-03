import numpy as np
import numpy.random as rd
import scipy.signal as sgn

from scpai.mz import compute_mz_circular_signal

INS_RESOLUTION = 10  # eV


def load_radiative_properties_file(fname: str):
    raw_data = np.loadtxt(fname).T

    keys = ["E", "bb", "bf", "ff", "emi", "obb", "obf", "off", "opa"]
    data = {}
    for key, row in zip(keys, raw_data[:-2]):
        data[key] = row

    # return data["E"], data["bb"] + data["bf"], data["opa"]  # bb + bf signal
    return data["E"], data["bb"], data["opa"]  # bb signal


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

        noise = rd.normal(loc=0, scale=self.fnoise * max(signal), size=len(self.egrid))
        signal += noise

        signal = apply_instrument_resolution(egrid, signal, INS_RESOLUTION / 2.355)
        return (signal - min(signal)) / (max(signal) - min(signal))


class Signal_MZ:
    def __init__(self, egrid: list, geometry: str, nzones: int, fnoise: float):
        self.egrid = egrid

        self.nzones = nzones
        self.fnoise = fnoise

        valid_geometries = ["S", "C"]
        if geometry not in valid_geometries:
            raise Exception(f"Geometry symbol {geometry} not recognized.")

        self.geometry = geometry

    def get_nsignal(self, fname_list: list, clength: float):
        if self.nzones != len(fname_list):
            raise ValueError(
                f"Number of zones {self.nzones} must coincide with file list size {len(fname_list)}"
            )

        egrid, j, k = np.array(
            [
                np.asarray(load_radiative_properties_file(fname)).T
                for fname in fname_list
            ]
        ).T

        j = np.array([np.interp(self.egrid, egrid.T[0], j_zone) for j_zone in j.T])
        k = np.array([np.interp(self.egrid, egrid.T[0], k_zone) for k_zone in k.T])

        if self.geometry == "C":
            signal = compute_mz_circular_signal(self.nzones, self.egrid, j, k, clength)

        signal = np.sum(signal, axis=0)
        noise = rd.normal(loc=0, scale=self.fnoise * max(signal), size=len(self.egrid))
        signal += noise

        signal = apply_instrument_resolution(egrid.T[0], signal, INS_RESOLUTION / 2.355)
        return (signal - min(signal)) / (max(signal) - min(signal))

    def get_contribution_by_zone(self, fname_list: list, clength: float):
        if self.nzones != len(fname_list):
            raise ValueError(
                f"Number of zones {self.nzones} must coincide with file list size {len(fname_list)}"
            )

        egrid, j, k = np.array(
            [
                np.asarray(load_radiative_properties_file(fname)).T
                for fname in fname_list
            ]
        ).T

        j = np.array([np.interp(self.egrid, egrid.T[0], j_zone) for j_zone in j.T])
        k = np.array([np.interp(self.egrid, egrid.T[0], k_zone) for k_zone in k.T])

        signal = compute_mz_circular_signal(self.nzones, self.egrid, j, k, clength)
        signal = apply_instrument_resolution(
            egrid.T[0], np.sum(signal, axis=0), INS_RESOLUTION / 2.355
        )

        signal_by_zone = np.empty((self.nzones, len(self.egrid)))
        for nzone in range(self.nzones):
            j_aux = j.copy()
            j_aux[nzone] = np.zeros_like(j[nzone])

            if self.geometry == "C":
                zone_signal = np.sum(
                    compute_mz_circular_signal(
                        self.nzones,
                        self.egrid,
                        j_aux,
                        k,
                        clength,
                    ),
                    axis=0,
                )

                zone_signal = apply_instrument_resolution(
                    egrid.T[0], zone_signal, INS_RESOLUTION / 2.355
                )

                signal_by_zone[nzone] = signal - zone_signal

        return signal_by_zone / np.max(signal)
