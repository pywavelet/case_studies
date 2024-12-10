from typing import Union

import numpy as np
from pywavelet.transforms.types import FrequencySeries, TimeSeries

from .random import rng


def CornishPowerSpectralDensity(f: np.ndarray) -> FrequencySeries:
    """
    PSD obtained from: https://arxiv.org/pdf/1803.01944.pdf
    Removed galactic confusion noise. Non stationary effect.
    """

    L = 2.5 * 10**9  # Length of LISA arm
    f0 = 19.09 * 10**-3

    Poms = ((1.5 * 10**-11) ** 2) * (
        1 + ((2 * 10**-3) / f) ** 4
    )  # Optical Metrology Sensor
    Pacc = (
        (3 * 10**-15) ** 2
        * (1 + (4 * 10**-3 / (10 * f)) ** 2)
        * (1 + (f / (8 * 10**-3)) ** 4)
    )  # Acceleration Noise

    PSD = (
        (10 / (3 * L**2))
        * (Poms + (4 * Pacc) / ((2 * np.pi * f)) ** 4)
        * (1 + 0.6 * (f / f0) ** 2)
    )  # PSD

    PSD[0] = PSD[1]
    return FrequencySeries(data=PSD, freq=f)


def noise_PSD_AE(f: np.ndarray, TDI="TDI1"):
    """
    Takes in frequency, spits out TDI1 or TDI2 A channel, same as E channel is equal and constant arm length approx.
    """
    L = 2.5e9
    c = 299758492
    x = 2 * np.pi * (L / c) * f

    Spm = (
        (3e-15) ** 2
        * (1 + ((4e-4) / f) ** 2)
        * (1 + (f / (8e-3)) ** 4)
        * (1 / (2 * np.pi * f)) ** 4
        * (2 * np.pi * f / c) ** 2
    )
    Sop = (15e-12) ** 2 * (1 + ((2e-3) / f) ** 4) * ((2 * np.pi * f) / c) ** 2

    S_val = 2 * Spm * (3 + 2 * np.cos(x) + np.cos(2 * x)) + Sop * (
        2 + np.cos(x)
    )

    if TDI == "TDI1":
        S = 8 * (np.sin(x) ** 2) * S_val
    elif TDI == "TDI2":
        S = 32 * np.sin(x) ** 2 * np.sin(2 * x) ** 2 * S_val
    else:
        raise ValueError("TDI must be either TDI1 or TDI2")
    return FrequencySeries(data=S, freq=f)


def generate_stationary_noise(
    ND: int, dt: float, psd: FrequencySeries, time_domain: bool = True
) -> Union[TimeSeries, FrequencySeries]:

    variance_f = (
        ND * psd.data / (4 * dt)
    )  # Variance of stationary noise process

    # variance_f[0] = variance_f[
    #     1
    # ]  # Set DC component to be the same as the next frequency


    # Generate noise in frequency domain
    real_noise_f = rng.normal(0, np.sqrt(variance_f))
    imag_noise_f = rng.normal(0, np.sqrt(variance_f))
    noise_f = real_noise_f + 1j * imag_noise_f

    # Real process (positive frequencies).
    noise_f[0] = np.sqrt(2) * noise_f[0].real
    noise_f[-1] = np.sqrt(2) * noise_f[-1].real

    noise_series = FrequencySeries(noise_f, psd.freq)

    # Convert to time-series
    if time_domain:
        return noise_series.to_timeseries()
    else:
        return noise_series
