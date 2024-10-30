import numpy as np
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from scipy.signal.windows import tukey
from typing import Tuple
import warnings

# Set filter to show each warning only once
warnings.simplefilter("once")

def inner_prod(sig1_f:np.ndarray, sig2_f:np.ndarray, PSD:np.ndarray, delta_t:float, N_t:int)->float:
    # Compute inner product. Useful for likelihood calculations and SNRs.
    return (4 * delta_t / N_t) * np.real(
        sum(np.conjugate(sig1_f) * sig2_f / PSD)
    )

def compute_snr_freq(h_f:np.ndarray, PSD:np.ndarray, delta_t:float, N:int)->float:
    """
    Compute the optimal matched filtering SNR in the frequency domain.

    Parameters:
    - h_f: Fourier transform of the signal.
    - PSD: Power spectral density.
    - delta_t: Sampling interval.
    - N: Length of time series.

    Returns:
    - snr: SNR value.
    """
    SNR2 = inner_prod(h_f, h_f, PSD, delta_t, N)
    return np.sqrt(SNR2)


def waveform(a:float, f:float, fdot:float, t:np.ndarray)->np.ndarray:
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """

    return a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2)))



def zero_pad(data:np.ndarray)->np.ndarray:
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    n_pad = int((2**pow_2) - N)
    # warn that we are padding the data (Log one time only in entire execution)
    if n_pad > 0:
        warnings.warn(f"Padding the data to a power of two. {N} -> {N + n_pad}. Avoid this by ensuring data in powers of 2.")
    return np.pad(data, (0, n_pad), "constant")


def waveform_generator(a:float, f:float, fdot:float, t:np.ndarray, tmax:float, alpha:float=0.0)->TimeSeries:
    """
    This function generates the waveform for a given set of parameters.
    """
    h_t = waveform(a, f, fdot, t)
    delta_t = t[1] - t[0]
    h_t_pad = zero_pad(h_t * tukey(len(h_t), alpha=alpha))
    t_pad = np.arange(0, len(h_t_pad) * delta_t, delta_t)
    # TODO: ask ollie if we can do this
    h_t_pad[t_pad > tmax] = 0

    return TimeSeries(
        data=h_t_pad,
        time=t_pad
    )


def generate_padded_signal(a:float, ln_f:float, ln_fdot:float, tmax:float, alpha:float=0)->Tuple[TimeSeries, FrequencySeries]:
    f = np.exp(ln_f)
    fdot = np.exp(ln_fdot)
    fs = 2 * f
    delta_t = np.floor(0.4 / fs)
    t = np.arange(0, tmax, delta_t)
    N = int(2 ** np.ceil(np.log2(len(t))))

    ht = waveform_generator(a, f, fdot, t, tmax, alpha, )
    hf = ht.to_frequencyseries()

    #TODO: ASK OLLIE :
    # h_true_f = np.fft.rfft(h_t_pad)
    # freq = np.fft.rfftfreq(N, delta_t)
    # freq[0] = freq[1] <-- IS THIS NECESSARY?

    return ht, hf