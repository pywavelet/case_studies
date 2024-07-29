import os
import sys
import numpy as np
from corner import corner
from scipy.signal.windows import tukey
from tqdm import tqdm

from numpy.random import normal

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import FrequencySeries
from pywavelet.transforms.to_wavelets import (from_freq_to_wavelet)
from pywavelet.transforms.from_wavelets import (from_wavelet_to_time)

from pywavelet.utils.lisa import waveform,  zero_pad

import matplotlib.pyplot as plt

sys.path.append("../Frequency_Domain/")
from gap_funcs import gap_routine, get_Cov, noise_PSD_AE
np.random.seed(1234)



def PowerSpectralDensity(f):
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

    return PSD


def __zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")


def FFT(waveform):
    """
    Here we taper the signal, pad and then compute the FFT. We remove the zeroth frequency bin because
    the PSD (for which the frequency domain waveform is used with) is undefined at f = 0.
    """
    N = len(waveform)
    taper = tukey(N, 0.1)
    waveform_w_pad = __zero_pad(waveform * taper)
    return np.fft.rfft(waveform_w_pad)


def freq_PSD(waveform_t, delta_t):
    """
    Here we take in a waveform and sample the correct fourier frequencies and output the PSD. There is no
    f = 0 frequency bin because the PSD is undefined there.
    """
    n_t = len(__zero_pad(waveform_t))
    freq = np.fft.rfftfreq(n_t, delta_t)
    freq[0] = freq[1] # redefining zeroth frequency to stop PSD -> infinity
    PSD = PowerSpectralDensity(freq)

    return freq, PSD


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


def llike(data_wavelet, signal_wavelet, psd_wavelet):
    """
    Computes log likelihood
    Assumption: Known PSD otherwise need additional term
    Inputs:
    data in frequency domain
    Proposed signal in frequency domain
    Variance of noise
    """
    inn_prod_wavelet = np.nansum(((data_wavelet - signal_wavelet) ** 2) / psd_wavelet)
    return -0.5 * inn_prod_wavelet

# Set true parameters. These are the parameters we want to estimate using MCMC.

a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

tmax = 10 * 60 * 60  # Final time
fs = 2 * f_true  # Sampling rate
delta_t = np.floor(
    0.1 / fs
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
PSD = noise_PSD_AE(freq, TDI = "TDI1")
# freq, PSD = freq_PSD(t, delta_t)  # Extract frequency bins and PSD.

SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))


# Compute things in the wavelet domain

signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
psd = FrequencySeries(data = PSD, freq = freq)

Nf = 64
Nt = N//Nf
print("Length of signal: ",N)
print("Wavelet bins in frequency= ",Nf)
print("Wavelet bins in time =  ",Nt)

kwgs = dict(
    Nf=Nf,
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

variance_noise_f = N * PSD/(4*delta_t)   # Compute variance in frequency domain (pos freq)

# Gaps in the frequency domain. 
start_window = 5
end_window = 7
lobe_length = 1.5
w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)

h_pad_w = w_t * h_t_pad 

# plt.plot(t_pad/60/60, h_pad_w);
# plt.plot(t_pad/60/60,np.max(h_pad_w)*w_t, label = 'Window', c='red')
# plt.xlabel(r'Time t [Hrs]')
# plt.ylabel(r'Signal')
# plt.title('Gaps')
# plt.grid()
# plt.savefig("plots/waveform.pdf",bbox_inches = "tight")
# plt.clf()
h_pad_fft = np.fft.rfft(h_pad_w) 

signal_gap_f = FrequencySeries(h_pad_fft, freq = freq)
h_wavelet_gap = Data.from_frequencyseries(signal_gap_f, **kwgs).wavelet

# Check, is this working?

h_t_gap_reconstructed = from_wavelet_to_time(h_wavelet_gap, delta_t)

# plt.plot(t_pad/60/60,h_pad_w,label = 'Truth') 
# plt.plot(t_pad/60/60,h_t_gap_reconstructed.data,label = 'reconstructed signal', linestyle = '--')
# plt.xlabel(r'Time [hours]')
# plt.ylabel(r'Amplitude')
# plt.title(r'Reconstruction')
# plt.show()



# ====================== ESTIMATE THE NOISE COVARIANCE MATRIX ==============================
print("Estimating the gated covariance matrix")
noise_wavelet_matrices = []
noise_gap_wavelet_matrices = []
for i in tqdm(range(0,500)):
    np.random.seed(i)
    noise_f_iter = np.random.normal(0,np.sqrt(variance_noise_f))  + 1j * np.random.normal(0,np.sqrt(variance_noise_f)) 
    noise_f_iter[0] = np.sqrt(2)*noise_f_iter[0].real
    noise_f_iter[-1] = np.sqrt(2)*noise_f_iter[-1].real

    noise_f_freq_series = FrequencySeries(noise_f_iter, freq = freq)    
    noise_wavelet = from_freq_to_wavelet(noise_f_freq_series, **kwgs)

    noise_t_iter = np.fft.irfft(noise_f_iter)      # Compute stationary noise in TD
    noise_t_gap_iter = w_t * noise_t_iter  # Place gaps in the noise from the TD
    noise_f_gap_iter = np.fft.rfft(noise_t_gap_iter) # Convert into FD 
    
    noise_f_freq_gap_series = FrequencySeries(noise_f_gap_iter, freq = freq)
    noise_gap_wavelet = from_freq_to_wavelet(noise_f_freq_gap_series, **kwgs)

    noise_wavelet_matrices.append(noise_wavelet.data)
    noise_gap_wavelet_matrices.append(noise_gap_wavelet.data)


# Convert list to 3D numpy array for easier manipulation
noise_gap_wavelet_matrix = np.array(noise_gap_wavelet_matrices)  # Shape: (1000, 32, 32)
noise_wavelet_matrix = np.array(noise_wavelet_matrices)  # Shape: (1000, 32, 32)

# Calculate the covariance matrix for each element in the 32x32 matrices
N_f = noise_gap_wavelet_matrix[0].data.shape[0]
N_t = noise_gap_wavelet_matrix[0].data.shape[1]

cov_matrix_wavelet = np.zeros((N_f,N_t), dtype = float)
cov_matrix_gap_wavelet = np.zeros((N_f,N_t), dtype=float)

for i in range(N_f):
    for j in range(N_t):
        cov_matrix_wavelet[i,j] = np.cov(noise_wavelet_matrix[:, i, j], rowvar=False)
        cov_matrix_gap_wavelet[i, j] = np.cov(noise_gap_wavelet_matrix[:, i, j], rowvar=False)


# add noise to wavelet coefficients, (easier to see)


noise = abs(np.random.normal(0,1e-30, (N_f,N_t)))
h_noise_no_gap = abs(h_wavelet.data) #+ noise
h_noise_gap = abs(h_wavelet_gap.data) #+ noise


# for i in range(N_f):
#     for j in range(N_t):
#         noise = np.random.normal(0,1e-24)
#         h_noise_no_gap[i,j] = h_wavelet.data[i,j] + noise
#         h_noise_gap[i,j] = h_wavelet_gap.data[i,j] + noise 
#         cov_matrix_gap_wavelet[i, j] = np.cov(noise_gap_wavelet_matrix[:, i, j], rowvar=False)

from pywavelet.plotting import plot_wavelet_grid

import matplotlib.pyplot as plt
fig,ax = plt.subplots(2,2, figsize = (16,8))

kwargs_cov_matrix = {"title":"Estimated wavelet covariance matrix"}
kwargs_h_matrix = {"title":"Signal wavelet matrix"}
kwargs_gap_wavelet_matrix = {"title":"Wavelet covariance matrix with gaps in data"}
kwargs_h_gap_matrix = {"title":"Signal wavelet matrix gaps"}

freq_range = [0,0.02]
plot_wavelet_grid(cov_matrix_wavelet,
                time_grid=psd_wavelet.time/60/60,
                freq_grid=psd_wavelet.freq,
                ax=ax[0,0],
                zscale="log",
                freq_scale="linear",
                absolute=False,
                freq_range = freq_range,
                **kwargs_cov_matrix)

plot_wavelet_grid(h_wavelet.data,
                time_grid=h_wavelet.time/60/60,
                freq_grid=h_wavelet.freq,
                ax=ax[0,1],
                zscale="linear",
                freq_scale="linear",
                absolute=False,
                freq_range = freq_range,
                **kwargs_h_matrix)

                
plot_wavelet_grid(cov_matrix_gap_wavelet,
                time_grid=noise_gap_wavelet.time/60/60,
                freq_grid=noise_gap_wavelet.freq,
                ax=ax[1,0],
                zscale="log",
                freq_scale="linear",
                absolute=False,
                freq_range = freq_range,
                **kwargs_gap_wavelet_matrix)


plot_wavelet_grid(h_wavelet_gap.data,
                time_grid=h_wavelet_gap.time/60/60,
                freq_grid=h_wavelet_gap.freq,
                ax=ax[1,1],
                zscale="linear",
                freq_scale="linear",
                absolute=False,
                freq_range = freq_range,
                **kwargs_h_gap_matrix)


plt.savefig("plots/Spectrogram_Nf_{}_Nt_{}_start_{}_end_{}_lobe_length_{}_tmax_{}.pdf".format(N_f, N_t, 4, 6, 1, tmax), bbox_inches = "tight")
plt.show()
plt.clf()

w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)

SNR2_estmated_gaps_wavelet = np.nansum((h_wavelet_gap*h_wavelet_gap) / cov_matrix_gap_wavelet)
SNR2_estmated_no_gaps_wavelet = np.nansum((h_wavelet*h_wavelet) / cov_matrix_wavelet)

print("estimated SNR with gaps using estimated gated wavelet covariance", SNR2_estmated_gaps_wavelet**(1/2))
print("estimated SNR with no gaps using estimated wavelet covariance", SNR2_estmated_no_gaps_wavelet**(1/2))
