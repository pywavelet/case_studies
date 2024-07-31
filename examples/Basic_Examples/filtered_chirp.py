import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
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

def bandpass_data(rawstrain, f_min_bp, f_max_bp, srate_dt, bandpassing_flag):
     """

    Bandpass the raw strain between [f_min, f_max] Hz.

    Arguments
    ---------

    rawstrain: numpy array
    The raw strain data.
    f_min_bp: float
    The lower frequency of the bandpass filter.
    f_max_bp: float
    The upper frequency of the bandpass filter.
    srate_dt: float
    The sampling rate of the data.
    bandpassing_flag: bool
    Whether to apply bandpassing or not.

    Returns
    -------

    strain: numpy array
    The bandpassed strain data.

    """
     if(bandpassing_flag):
     # Bandpassing section.
         # Create a fourth order Butterworth bandpass filter between [f_min, f_max] and apply it with the function filtfilt.
         bb, ab = butter(2, [f_min_bp/(0.5*srate_dt), f_max_bp/(0.5*srate_dt)], btype='band')
         strain = filtfilt(bb, ab, rawstrain)
     else:
         strain = rawstrain

     return strain
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

# Let's now try to filter and see what happens 
low_pass = 50
high_pass = 300
filtered_y = bandpass_data(y, low_pass, high_pass, 1/dt, True)

filtered_y_fft = np.fft.rfft(filtered_y)

fig, ax = plt.subplots(2, 2, figsize=(16, 5))
ax[0,0].plot(t, y)
ax[0,0].set_xlabel('Time [seconds]')
ax[0,0].set_ylabel('Strain')
ax[0,0].set_title('Chirping Sinusoid')
ax[0,0].grid()

ax[0,1].loglog(freq, abs(y_fft)**2)
ax[0,1].set_xlabel('Frequency [Hz]')
ax[0,1].set_ylabel('Periodigram')
ax[0,1].set_title('Frequency domain')
ax[0,0].grid()

ax[1,0].plot(t, filtered_y)
ax[1,0].set_xlabel('Time [seconds]')
ax[1,0].set_ylabel('Strain')
ax[1,0].set_title('Chirping Sinusoid, filtered')
ax[1,0].grid()

ax[1,1].loglog(freq, abs(filtered_y_fft)**2)
ax[1,1].set_xlabel('Time [seconds]')
ax[1,1].set_ylabel('Strain')
ax[1,1].set_title('Chirping Sinusoid, filtered')
ax[1,1].grid()
plt.tight_layout()
plt.show()
plt.clf()

# Check overlaps

fig, ax = plt.subplots(1,2, figsize = (16,5))

ax[0].plot(t,y, label = 'Truth')
ax[0].plot(t,filtered_y, label = 'Filtered', c = 'red', linestyle = '--')
ax[0].set_xlabel("time")
ax[0].set_ylabel("amplitude")
ax[0].set_title("Filtered time series")

ax[1].loglog(freq,abs(y_fft)**2, label = 'Truth')
ax[1].loglog(freq,abs(filtered_y_fft)**2, label = 'Filtered', linestyle = '--', c = 'red')
ax[1].set_xlabel("time")
ax[1].set_ylabel("amplitude")
ax[1].set_title("Filtered freq series")
ax[1].axvline(low_pass, c = 'black', linestyle = '--', label = 'low frequency filter')
ax[1].axvline(high_pass, c = 'black', linestyle = '--', label = 'high frequency filter')
ax[1].legend()
plt.tight_layout()
plt.show()

# quit()


# Power spectral density
PSD = PSD_AMP * np.ones(len(freq))

# Compute SNR in frequency domain
SNR2_f = 4 * dt * np.sum(abs(y_fft) ** 2 / (ND * PSD))

########################################
# Part 2: Wavelet domain
########################################

# Convert to wavelet domain
signal_frequencyseries = FrequencySeries(y_fft, freq=freq)
signal_frequencyseries_filtered = FrequencySeries(filtered_y_fft, freq=freq)

signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf=Nf)
signal_wavelet_freq_filtered = from_freq_to_wavelet(signal_frequencyseries_filtered, Nf=Nf)
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
snr2_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2
snr2_freq_to_wavelet_filtered = compute_snr(signal_wavelet_freq_filtered, psd_wavelet_freq) ** 2

# Print SNR results
print("SNR in frequency domain", SNR2_f**0.5)
print("SNR using wavelet domain, freq -> wavelet", snr2_freq_to_wavelet**0.5)
print("SNR using wavelet domain, freq filtered -> wavelet", snr2_freq_to_wavelet_filtered**0.5)


# Plot wavelet grid
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
plot_wavelet_grid(
    signal_wavelet_freq.data,
    time_grid=signal_wavelet_freq.time,
    freq_grid=signal_wavelet_freq.freq,
    ax=ax[0],
    zscale="linear",
    freq_scale="linear",
    absolute=False,
    freq_range=[0, 300]
)

ax[0].set_xlabel('Wavelet time bins')  # Custom x-label for the first subplot
ax[0].set_ylabel('Wavelet Frequency bins')  # Custom y-label for the first subplot
ax[0].set_title('Chirping sinusoid')  # Custom title for the first subplot

fig = plot_wavelet_grid(
    signal_wavelet_freq_filtered.data,
    time_grid=signal_wavelet_freq.time,
    freq_grid=signal_wavelet_freq.freq,
    ax=ax[1],
    zscale="linear",
    freq_scale="linear",
    absolute=False,
    freq_range=[0, 300]
)

ax[1].set_xlabel('Wavelet time bins')  # Custom x-label for the first subplot
ax[1].set_ylabel('Wavelet Frequency bins')  # Custom y-label for the first subplot
ax[1].set_title('Filtered chirping sinusoid')  # Custom title for the first subplot
plt.show()
plt.clf()
breakpoint()
quit()
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
