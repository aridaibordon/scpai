import numpy as np

from itertools import chain


def get_chord_length(clength, zone, subzone, nzones):
    if zone == subzone:
        return (
            (2 * subzone + 1) ** 2
            * clength
            / nzones
            * (
                np.pi / 4
                - (1 / 2) * np.arcsin((2 * subzone - 1) / (2 * subzone + 1))
                - (1 / 2)
                * (
                    (2 * subzone - 1)
                    / (2 * subzone + 1)
                    * np.sqrt(1 - ((2 * subzone - 1) / (2 * subzone + 1)) ** 2)
                )
            )
        )
    else:
        return (
            (
                (1 / 4) * ((2 * subzone + 1) ** 2 / (2 * subzone - 1))
                + (1 / 4) * np.sqrt(2 * subzone)
                - np.pi / 8 * (2 * subzone - 1)
            )
            * clength
            / nzones
        )


def get_atenuated_source(j, k, clength, zone, subzone, nzones):
    chord_length = get_chord_length(clength, zone, subzone, nzones)
    return j / k * (1 - np.exp(-k * chord_length))


def get_atenuation_factor(j, k, clength, zone, subzone, nzones):
    chord_length = get_chord_length(clength, zone, subzone, nzones)
    return np.exp(-k * chord_length)


def compute_mz_circular_signal(
    nzones: int, egrid: list, j: list, k: list, clength: float
):
    mz_signal = np.empty(nzones, dtype=object)
    for zone in range(nzones):
        zone_signal = np.zeros(len(egrid))
        for subzone in range(zone, nzones):
            if subzone == zone:
                zone_signal += get_atenuated_source(
                    j[subzone], k[subzone], clength, zone, subzone, nzones
                ) * np.prod(
                    [
                        get_atenuation_factor(
                            j[aux_zone], k[aux_zone], clength, aux_zone, subzone, nzones
                        )
                        for aux_zone in range(subzone + 1, nzones)
                    ],
                    axis=0,
                )
                continue

            zone_signal += get_atenuated_source(
                j[subzone], k[subzone], clength, zone, subzone, nzones
            ) * np.prod(
                [
                    get_atenuation_factor(
                        j[aux_zone], k[aux_zone], clength, aux_zone, subzone, nzones
                    )
                    for aux_zone in chain(range(zone, subzone), range(subzone, nzones))
                ],
                axis=0,
            )  # estrambótico
            zone_signal += get_atenuated_source(
                j[subzone], k[subzone], clength, zone, subzone, nzones
            ) * np.prod(
                [
                    get_atenuation_factor(
                        j[aux_zone], k[aux_zone], clength, aux_zone, subzone, nzones
                    )
                    for aux_zone in range(subzone + 1, nzones)
                ],
                axis=0,
            )

        mz_signal[zone] = np.asarray(zone_signal)

    return mz_signal
