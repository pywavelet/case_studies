import numpy as np
from pywavelet.transforms.types import TimeSeries


def waveform(a: float, f: float, fdot: float, t: np.ndarray) -> TimeSeries:
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    return TimeSeries(
        a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t ** 2))),
        t
    )


def waveform_generator(a: float, f: float, fdot: float, t: np.ndarray, tmax: float, alpha: float = 0.0) -> TimeSeries:
    ht = waveform(
        a, f, fdot, t
    ).zero_pad_to_power_of_2(tukey_window_alpha=alpha)
    ht.data[ht.time > tmax] = 0
    return ht
