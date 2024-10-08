import os

import numpy as np
from scipy.signal.windows import tukey
from tqdm import tqdm

from numpy.random import normal

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import FrequencySeries
from pywavelet.transforms.to_wavelets import (from_freq_to_wavelet)

from pywavelet.plotting import plot_wavelet_grid
import matplotlib.pyplot as plt

import sys
sys.path.append("../Frequency_Domain/")
from noise_curves import noise_PSD_AE


np.random.seed(1234)


def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")

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


a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

TDI = "TDI1"

tmax = 10 * 60 * 60  # Final time
fs = 2 * f_true  # Sampling rate
delta_t = np.floor(
    0.4 / fs
)  # Sampling interval -- largely oversampling here.

t = np.arange(
    0, tmax, delta_t
)  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N = int(
    2 ** (np.ceil(np.log2(len(t))))
)  # Round length of time series to a power of two.
# Length of time series
h_t = waveform(a_true, f_true, fdot_true, t)
taper_signal = tukey(len(h_t), alpha = 0.2)
h_t_pad = zero_pad(h_t*taper_signal)

t_pad = np.arange(0,len(h_t_pad)*delta_t, delta_t)

h_true_f = np.fft.rfft(h_t_pad)
freq = np.fft.rfftfreq(N, delta_t); freq[0] = freq[1]
PSD = noise_PSD_AE(freq, TDI = TDI)

SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))

#  =============== WAVELET DOMAIN ========================
signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
psd = FrequencySeries(data = PSD, freq = freq)


kwgs = dict(
    Nf=32,
)


h_wavelet = Data.from_frequencyseries(signal_f_series, **kwgs).wavelet
psd_wavelet = evolutionary_psd_from_stationary_psd(
                                                    psd=psd.data,
                                                    psd_f=psd.freq,
                                                    f_grid=h_wavelet.freq,
                                                    t_grid=h_wavelet.time,
                                                    dt=delta_t,
                                                )

Wavelet_Matrix = psd_wavelet.data

SNR2_wavelet = np.nansum((h_wavelet*h_wavelet) / psd_wavelet)
print("SNR in wavelet domain is", SNR2_wavelet**(1/2))


# ====================== ESTIMATE THE NOISE COVARIANCE MATRIX ==============================

variance_noise_f = N * PSD/(4*delta_t)   # Compute variance in frequency domain (pos freq)
print("Estimating the gated covariance matrix")
noise_wavelet_matrices = []
for i in tqdm(range(0,20000)):
    np.random.seed(i)
    noise_f_iter = np.random.normal(0,np.sqrt(variance_noise_f))  + 1j * np.random.normal(0,np.sqrt(variance_noise_f)) 
    noise_f_iter[0] = np.sqrt(2)*noise_f_iter[0].real
    noise_f_iter[-1] = np.sqrt(2)*noise_f_iter[-1].real
 
    noise_f_freq_series = FrequencySeries(noise_f_iter, freq = freq)
    noise_wavelet = from_freq_to_wavelet(noise_f_freq_series, **kwgs)
    noise_wavelet_matrices.append(noise_wavelet.data)

# Convert list to 3D numpy array for easier manipulation
noise_wavelet_matrix = np.array(noise_wavelet_matrices)  # Shape: (1000, 32, 32)

# Calculate the covariance matrix for each element in the 32x32 matrices
N_f = noise_wavelet.data.shape[0]
N_t = noise_wavelet.data.shape[1]

cov_matrix_wavelet = np.zeros((N_f,N_t),dtype = float)
for i in range(N_f):
    for j in range(N_t):
        cov_matrix_wavelet[i, j] = np.cov(noise_wavelet_matrix[:, i, j], rowvar=False)

fig,ax = plt.subplots(2,1)

plot_wavelet_grid(cov_matrix_wavelet,
                time_grid=noise_wavelet.time,
                freq_grid=noise_wavelet.freq,
                ax=ax[0],
                zscale="log",
                freq_scale="linear",
                absolute=False,
                freq_range = [noise_wavelet.freq[1], 5e-3])

plot_wavelet_grid(psd_wavelet.data,
                time_grid=psd_wavelet.time,
                freq_grid=psd_wavelet.freq,
                ax=ax[1],
                zscale="log",
                freq_scale="linear",
                absolute=False,
                freq_range = [psd_wavelet.freq[1], 5e-3])
plt.show()
plt.clf()

SNR2_estmated_wavelet = np.nansum((h_wavelet*h_wavelet) / cov_matrix_wavelet)
print("estimated SNR in using wavelet covariance", SNR2_estmated_wavelet**(1/2))

breakpoint()





