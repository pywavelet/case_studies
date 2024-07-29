import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import chirp
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.to_wavelets import from_time_to_wavelet, from_freq_to_wavelet
from pywavelet.transforms.from_wavelets import (
    from_wavelet_to_time, 
    from_wavelet_to_freq, 
)
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils.snr import compute_snr
from pywavelet.plotting import plot_wavelet_grid

def inner_prod(sig1_f, sig2_f, N, dt):
    return (sig1_f.conj() * sig2_f)

sys.path.append("../../../")

# Parameters
f0 = 0
f1 = 300
T = 10
A = 3.0
PSD_AMP = 1e-2

# Sampling parameters
dt = 0.01 / (2 * f1)  
t = np.arange(0, T, dt)  # Time array
t = t[:2 ** int(np.log2(len(t)))]  # Round len(t) to the nearest power of 2
T = len(t) * dt
ND = len(t)

Nf = 2**(13) # Set the number of wavelet frequency bins. 

# Generate chirp signal
# This will generate a signal starting at f0 at t=0 and ending with
# frequency f1 at time t = T
y = A*chirp(t, f0, T, f1, method='quadratic', phi=0, vertex_zero=True)

# Apply a window. You can really see the leakage if alpha = 0.2
alpha = 0.2 
window = tukey(len(y), alpha) 
y *= window

# Frequency array and FFT, positive transform + frequencies
freq = np.fft.rfftfreq(len(y), dt)
freq[0] = freq[1]
df = abs(freq[1] - freq[0])
y_fft = np.fft.rfft(y)

# Plot time domain and frequency domain
fig, ax = plt.subplots(1, 2, figsize=(16, 5))
ax[0].plot(t, y)
ax[0].set_xlabel('Time [seconds]')
ax[0].set_ylabel('Strain')
ax[0].set_title('Chirping Sinusoid')
ax[0].grid()

ax[1].loglog(freq, abs(y_fft)**2)
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Periodigram')
ax[1].set_title('Frequency domain')
ax[1].grid()

plt.show()
plt.clf()

# Power spectral density
PSD = PSD_AMP * np.ones(len(freq))

# Compute SNR in frequency domain
SNR2_f = 4 * dt * np.sum(abs(y_fft) ** 2 / (ND * PSD))

########################################
# Part 2: Wavelet domain
########################################

# Convert to wavelet domain
signal_timeseries = TimeSeries(y, t)
signal_frequencyseries = FrequencySeries(y_fft, freq=freq)

signal_wavelet_time = from_time_to_wavelet(signal_timeseries, Nf=Nf)
psd_wavelet_time = evolutionary_psd_from_stationary_psd(
    psd=PSD,
    psd_f=freq,
    f_grid=signal_wavelet_time.freq,
    t_grid=signal_wavelet_time.time,
    dt=dt,
)

signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf=Nf)
psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
    psd=PSD,
    psd_f=freq,
    f_grid=signal_wavelet_freq.freq,
    t_grid=signal_wavelet_freq.time,
    dt=dt,
)

# Compute SNR in wavelet domain
# here we compute from  time -> wavelet
# and freq -> wavelet
snr2_time_to_wavelet = compute_snr(signal_wavelet_time, psd_wavelet_time) ** 2
snr2_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2

# Print SNR results
print("SNR in frequency domain", SNR2_f**0.5)
print("SNR using wavelet domain, time -> wavelet", snr2_time_to_wavelet**0.5)
print("SNR using wavelet domain, freq -> wavelet", snr2_freq_to_wavelet**0.5)

# Plot wavelet grid
fig, ax = plt.subplots(1, 1, figsize=(16, 7))
plot_wavelet_grid(
    signal_wavelet_freq.data,
    time_grid=signal_wavelet_freq.time,
    freq_grid=signal_wavelet_freq.freq,
    ax=ax,
    zscale="linear",
    freq_scale="linear",
    absolute=False,
    freq_range=[0, 300]
)
plt.show()
plt.clf()

# Convert back from wavelet to frequency and time domains
# going to check signal reconstruction
signal_freq_from_wavelet = from_wavelet_to_freq(signal_wavelet_freq, dt)
signal_time_from_wavelet = from_wavelet_to_time(signal_wavelet_time, dt)

# Compute non-noiseweighted inner product between signals in both
# domains
ab_freq = np.sum(signal_freq_from_wavelet.data.conj() * y_fft)
aa_freq = np.sum(signal_freq_from_wavelet.data.conj() * signal_freq_from_wavelet.data)
bb_freq = np.sum(y_fft.conj() * y_fft)
overlap_freq = np.real(ab_freq/np.sqrt(aa_freq*bb_freq))

ab_time = np.sum(signal_time_from_wavelet.data.conj() * y)
aa_time = np.sum(signal_time_from_wavelet.data.conj() * signal_time_from_wavelet.data.conj()) 
bb_time = np.sum(abs(y * y))
overlap_time = ab_time/np.sqrt((aa_time * bb_time))

print("Overlap in the frequency domain is",overlap_freq)
print("Overlap in the time domain is",overlap_time)

# Plot the signal and reconstructed signal in each domain 
# on top of eachother
fig, ax = plt.subplots(2,2, figsize = (16,7))
for i in range(0,2):
    for j in range(0,2):
        ax[i,j].plot(t, y, label = 'truth', c = 'blue')
        ax[i,j].plot(t, signal_time_from_wavelet.data, c = 'red',linestyle = '--', label = 'approx')
        ax[i,j].set_xlabel(r'Time [seconds]')
        ax[i,j].set_ylabel(r'Amplitude [seconds]')
        ax[i,j].set_title(r'Comparing reconstructions')
        ax[i,j].legend()

ax[0,0].set_xlim([0,1.5])
ax[0,1].set_xlim([1.9,2.5])
ax[1,0].set_xlim([5,5.05])
ax[1,1].set_xlim([5.43,5.46])

plt.grid()
plt.tight_layout()
plt.show()
plt.clf()


# Plot reconstructed signal in frequency domain on top of true signal
# in frequency domain
fig, ax = plt.subplots(1,1, figsize = (16,7))
ax.loglog(freq, abs(y_fft)**2, c = 'blue', label = 'truth')
ax.loglog(freq, abs(signal_freq_from_wavelet.data)**2, c = 'red', linestyle = '--', label = 'approx')
ax.set_xlabel(r'Freq [Hz]')
ax.set_ylabel(r'Periodigram')
ax.set_title(r'Comparing reconstructions -- Freq Domain')

plt.tight_layout()
plt.grid()
plt.show()

# Compute residuals
res_freq = abs(signal_freq_from_wavelet.data - y_fft)**2
res_time = abs(signal_time_from_wavelet.data - y)**2

# Plot residuals
fig, ax = plt.subplots(1, 2, figsize=(16, 7))

ax[0].loglog(freq, res_freq)
ax[0].set_xlabel('Frequency')
ax[0].set_ylabel('Residuals (squared)')
ax[0].set_title('Frequency domain')

ax[1].plot(t, res_time)
ax[1].set_xlabel('Time seconds')
ax[1].set_ylabel('Residuals')
ax[1].set_title('Time domain')

plt.show()
plt.clf()
quit()
