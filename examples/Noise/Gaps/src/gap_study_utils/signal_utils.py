import numpy as np
from pywavelet.transforms.types import TimeSeries


def waveform(
    a: float, ln_f: float, ln_fdot: float, t: np.ndarray
) -> TimeSeries:
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """
    f, fdot = np.exp(ln_f), np.exp(ln_fdot)
    return TimeSeries(
        a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2))), t
    )
