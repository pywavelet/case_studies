import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (butter, filtfilt, freqz, sosfilt, sosfreqz,
                          freqz_zpk)
from scipy.signal.windows import tukey
from scipy.signal import chirp

# Define the bandpass_data function
def bandpass_data_ba(rawstrain, f_min_bp, fs, bandpassing_flag=False, order=4):
    if bandpassing_flag:
        f_nyq = 0.5 * fs
        b, a = butter(order, f_min_bp / f_nyq, btype='highpass', output = 'ba')
        strain = filtfilt(b, a, rawstrain)
    else:
        strain = rawstrain
    return strain

def bandpass_data_sos(rawstrain, f_min_bp, fs, bandpassing_flag=False, order=4):
    if bandpassing_flag:
        f_nyq = 0.5 * fs
        sos = butter(order, f_min_bp / f_nyq, btype='highpass', output = 'sos')
        strain = sosfilt(sos, rawstrain)
    else:
        strain = rawstrain
    return strain

# Define the get_frequency_response function
def get_frequency_response_ba(f_min_bp, fs, freqs, order=4):
    # N_freq = N // 2 + 1
    f_nyq = 0.5 * fs

    b, a = butter(order, f_min_bp / f_nyq, btype='highpass', output = 'ba')
    
    # Compute the frequency response
    w, h = freqz(b, a, worN=freqs * 2 * np.pi / fs)
    
    # Convert w to actual frequency in Hz
    freqs = w * fs / (2 * np.pi)
    
    return freqs, h

def get_frequency_response_sos(f_min_bp, fs, freqs, order=4):
    # N_freq = N // 2 + 1
    f_nyq = 0.5 * fs

    sos = butter(order, f_min_bp / f_nyq, btype='highpass', output = 'sos')
    
    # Compute the frequency response
    w, h = sosfreqz(sos, worN=freqs * 2 * np.pi / fs, whole=False, fs=fs)
    
    # Convert w to actual frequency in Hz
    freqs = w * fs / (2 * np.pi)
    
    return freqs, h
# Define the get_frequency_response function
def get_frequency_response_zpk(f_min_bp, fs, freqs, order=4):
    # N_freq = N // 2 + 1
    f_nyq = 0.5 * fs

    z,p,k = butter(order, f_min_bp / f_nyq, btype='highpass', output = 'zpk')
    
    # Compute the frequency response
    w, h = freqz_zpk(z,p,k, worN=freqs * 2 * np.pi / fs, whole=False, fs=fs)
    
    # Convert w to actual frequency in Hz
    freqs = w * fs / (2 * np.pi)
    
    return freqs, h

# Parameters
f0 = 50
f1 = 300
T = 10
A = 3.0
PSD_AMP = 1e-2

# Sampling parameters
dt = 0.01 / (2 * f1)  
fs = 1/dt
f_nyq = 0.5 * fs

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
N = len(y)
freq = np.fft.rfftfreq(len(y), dt)

# Determine the frequency index
f_low_index = np.argwhere(freq >= 100)[0][0]

# Set up filters
order = 6

b, a = butter(order, freq[f_low_index] / f_nyq, btype='highpass', output = 'ba')
sos = butter(order,  freq[f_low_index] / f_nyq, btype='highpass', output = 'sos')
z,p,k = butter(order,  freq[f_low_index] / f_nyq, btype='highpass', output = 'zpk')

# Apply the filters
y_filtered_ba = filtfilt(b, a, y)
y_filtered_sos = sosfilt(sos, y)
# Compute the frequency response
# freqs_ba, response_f_ba, = freqz(b, a, worN= freq * 2 * np.pi / fs, whole = False, fs = fs)
freqs_ba, response_f_ba, = freqz(b, a, worN = N//2 + 1 , whole = False, fs = fs)
freqs_sos, response_f_sos = sosfreqz(sos, worN= N//2 + 1, whole=False, fs=fs)
freqs_zpk, response_f_zpk = freqz_zpk(z,p,k, worN= N//2 + 1 , whole=False, fs=fs) 

# Apply the frequency response to the window function in the frequency domain

check_filter_ba = response_f_ba * np.fft.rfft(y)
check_filter_sos = response_f_sos * np.fft.rfft(y)
check_filter_zpk = response_f_zpk * np.fft.rfft(y)

# Compute the FFT of the filtered window function
filtered_w_fft_ba = np.fft.rfft(y_filtered_ba)
filtered_w_fft_sos = np.fft.rfft(y_filtered_sos)

breakpoint()
# Normalize frequency response multiplication
fig, ax = plt.subplots(1,3, figsize=(18, 6))
ax[0].loglog(freq, np.abs(check_filter_ba)**2, c = 'blue', label='K(f) * W(f)')
ax[0].loglog(freq, np.abs(filtered_w_fft_ba)**2, linestyle = '--', c = 'red', label='Filtered w(t)')
ax[0].loglog(freq, np.abs(response_f_ba)**2, label = 'Filter response')
ax[0].legend(loc = 'lower left')
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('Power')
ax[0].set_title('Using standard butter with (b,a) format')
ax[0].grid()
# Normalize frequency response multiplication
ax[1].loglog(freq, np.abs(check_filter_sos)**2, c = 'blue', label='K(f) * W(f)')
ax[1].loglog(freq, np.abs(filtered_w_fft_sos)**2, c = 'red', linestyle = '--', label='Filtered w(t)')

ax[1].loglog(freq, np.abs(response_f_sos)**2, label = 'Filter response')
ax[1].legend(loc='lower left')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Power')
ax[1].set_title('Using standard butter with sos format')
ax[1].grid()

ax[2].loglog(freq, np.abs(check_filter_zpk)**2, c = 'blue', label='K(f) * W(f)')
ax[2].loglog(freq, np.abs(filtered_w_fft_sos)**2, c = 'red', linestyle = '--', label='Filtered w(t)')
ax[2].loglog(freq, np.abs(response_f_zpk)**2, label = 'Filter response')
ax[2].legend(loc='lower left')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('Power')
ax[2].set_title('Using standard butter with zpk format')
ax[2].grid()
plt.show()