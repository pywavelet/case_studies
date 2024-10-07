from tkinter import W
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal.windows import tukey
# from pywavelet.data import Data
from pywavelet.utils import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.to_wavelets import from_time_to_wavelet, from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr
# from pywavelet.plotting import plot_wavelet_grid

f0 = 20
T = 1000
A = 2.0
PSD_AMP = 1e-2
Nf = 512

dt = 0.5 / (2 * f0)  # Shannon's sampling theorem, set dt < 1/2*highest_freq
t = np.arange(0, T, dt)  # Time array
# round len(t) to the nearest power of 2
t = t[: 2 ** int(np.log2(len(t)))]
T = len(t) * dt
ND = len(t)

y = A * np.sin(2 * np.pi * f0 * t)  # Signal waveform we wish to test
window = tukey(len(y), 0.0) # Essential if you want to remove leakage
y *= window # Apply taper
freq = np.fft.rfftfreq(len(y), dt)  # Frequencies
df = abs(freq[1] - freq[0])  # Sample spacing in frequency
y_fft = np.fft.rfft(y)  # continuous time fourier transform [seconds]


PSD = PSD_AMP * np.ones(len(freq))  # PSD of the noise

# Compute the SNRs
SNR2_f = 4 * dt * np.sum(abs(y_fft) ** 2 / (ND*PSD))
SNR2_t = 2 * dt * np.sum(abs(y) ** 2 / PSD_AMP)
SNR2_t_analytical = (A**2) * T / PSD[0]

########################################
# Part2: Wavelet domain
########################################

Nt = ND // Nf
signal_timeseries = TimeSeries(y, t)
signal_frequencyseries = FrequencySeries(y_fft,freq = freq)

# From time to wavelet
signal_wavelet_time = from_time_to_wavelet(signal_timeseries, Nf = Nf)
psd_wavelet_time= evolutionary_psd_from_stationary_psd(
    psd=PSD,
    psd_f=freq,
    f_grid=signal_wavelet_time.freq,
    t_grid=signal_wavelet_time.time,
    dt=dt,
)

# From frequency to wavelet
signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf = Nf)
psd_wavelet_freq= evolutionary_psd_from_stationary_psd(
    psd=PSD,
    psd_f=freq,
    f_grid=signal_wavelet_freq.freq,
    t_grid=signal_wavelet_freq.time,
    dt=dt,
)

# Compute SNR in wavelet domain
snr2_time_to_wavelet = compute_snr(signal_wavelet_time, psd_wavelet_time) ** 2
snr2_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2

# print out all the different SNRs 
print("SNR in frequency domain", SNR2_f**(1/2))
print("SNR in time domain (parseval's theorem)", SNR2_t**(1/2))
print("SNR using analytical formulas", SNR2_f**(1/2))
print("SNR using wavelet domain, time -> wavelet", snr2_time_to_wavelet**(1/2))
print("SNR using wavelet domain, freq -> wavelet", snr2_freq_to_wavelet**(1/2))

# Does this actually make sense?


# fig,ax = plt.subplots(2,1)
# plot_wavelet_grid(signal_wavelet_freq.data,
#                 time_grid=signal_wavelet_freq.time,
#                 freq_grid=signal_wavelet_freq.freq,
#                 ax=None,
#                 zscale="linear",
#                 freq_scale="linear",
#                 absolute=False,
#                 freq_range=[18,22])
# plt.show()
# plt.clf()

print("Maximum value at f = {} is {} using freq -> wavelet".format(f0, max(signal_wavelet_freq.data.flatten())))
print("Maximum value at f = {} is {} using time -> wavelet".format(f0, max(signal_wavelet_time.data.flatten())))
# Ollie's analytical guess
print("Hypothesis for true wavelet domain transformation is {}".format(A*np.sqrt(2*Nf)))

