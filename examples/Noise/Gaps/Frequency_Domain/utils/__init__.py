import numpy as np

def inner_prod(sig1_f, sig2_f, PSD, delta_t, N_t):
    # Compute inner product. Useful for likelihood calculations and SNRs.
    return (4 * delta_t / N_t) * np.real(
        sum(np.conjugate(sig1_f) * sig2_f / PSD)
    )



def waveform(a, f, fdot, t, eps=0):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """

    return a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2)))



def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


