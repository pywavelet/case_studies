import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (butter, filtfilt, freqz, sosfilt, sosfiltfilt, sosfreqz,
                          freqz_zpk, tf2zpk, sos2zpk)
from scipy.signal.windows import tukey
from scipy.signal import chirp

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
order = 15

# Standard (b,a) format. Not great for high orders. Numerically unstable. 
b, a = butter(order, freq[f_low_index] / f_nyq, btype='highpass', output = 'ba')
z_from_ab, p_from_ab, k_from_ab = tf2zpk(b,a)
print("Check the poles of coefficients of b, a in butter")
print(np.sort(abs(p_from_ab)))

# second order cascade format. Much more stable. 
sos = butter(order,  freq[f_low_index] / f_nyq, btype='highpass', output = 'sos')
z_from_sos, p_from_sos, k_from_sos = sos2zpk(sos)
print("Now, from the second order cascading method (sos), check poles")
print(np.sort(abs(p_from_sos)))

# Another method, very stable. 
z,p,k = butter(order,  freq[f_low_index] / f_nyq, btype='highpass', output = 'zpk')
print("converting straight to zpk format gives poles:")
print(np.sort(abs(p)))

# Apply the filters, forward and backward filters
y_filtered_ba = filtfilt(b, a, y)
y_filtered_sos = sosfiltfilt(sos, y)

# Compute the frequency response
freqs_ba, response_f_ba, = freqz(b, a, worN = N//2 + 1 , whole = False, fs = fs)
freqs_sos, response_f_sos = sosfreqz(sos, worN= N//2 + 1, whole=False, fs=fs)
freqs_zpk, response_f_zpk = freqz_zpk(z,p,k, worN= N//2 + 1 , whole=False, fs=fs) 

breakpoint()
# Apply the frequency response to the window function in the frequency domain
# If we use filtfilt then we filter forwards and then backwards in time. This
# removes any phase offset. The true response function in freq domain is the
# squared response abs(response_f)**2. 
# check_filter_ba = abs(response_f_ba) * np.fft.rfft(y)
check_filter_ba = abs(response_f_ba)**2 * np.fft.rfft(y)
check_filter_sos = abs(response_f_sos)**2 * np.fft.rfft(y)
check_filter_zpk = abs(response_f_zpk)**2  * np.fft.rfft(y)
# If we filter just forwards, then we just need response_f_ba * ...

# Compute the FFT of the filtered window function
filtered_y_fft_ba = np.fft.rfft(y_filtered_ba)
filtered_y_fft_sos = np.fft.rfft(y_filtered_sos)

# Normalize frequency response multiplication
fig, ax = plt.subplots(1,3, figsize=(18, 6), sharey = True)
ax[0].loglog(freq, np.abs(check_filter_ba)**2, c = 'blue', label='K(f) * W(f)')
ax[0].loglog(freq, np.abs(filtered_y_fft_ba)**2, linestyle = '--', c = 'red', label='Filtered w(t)')
ax[0].loglog(freq, np.abs(response_f_ba)**2, label = 'Filter response')
ax[0].legend(loc = 'lower left')
ax[0].set_xlabel('Frequency [Hz]')
ax[0].set_ylabel('Power')
ax[0].set_ylim([1e-20,1e10])
ax[0].set_title('Using butter filtfilt with (b,a) format at order {}'.format(order), fontsize = 10)
ax[0].grid()
# Normalize frequency response multiplication
ax[1].loglog(freq, np.abs(check_filter_sos)**2, c = 'blue', label='K(f) * W(f)')
ax[1].loglog(freq, np.abs(filtered_y_fft_sos)**2, c = 'red', linestyle = '--', label='Filtered w(t)')

ax[1].loglog(freq, np.abs(response_f_sos)**2, label = 'Filter response')
ax[1].legend(loc='lower left')
ax[1].set_xlabel('Frequency [Hz]')
ax[1].set_ylabel('Power')
ax[1].set_title('Using butter with sosfiltfilt format at order {}'.format(order), fontsize = 10)
ax[1].grid()

ax[2].loglog(freq, np.abs(check_filter_zpk)**2, c = 'blue', label='K(f) * W(f)')
ax[2].loglog(freq, np.abs(filtered_y_fft_sos)**2, c = 'red', linestyle = '--', label='Filtered w(t)')
ax[2].loglog(freq, np.abs(response_f_zpk)**2, label = 'Filter response')
ax[2].legend(loc='lower left')
ax[2].set_xlabel('Frequency [Hz]')
ax[2].set_ylabel('Power')
ax[2].set_title('Using butter sosfiltfilt with zpk format at order {}'.format(order), fontsize = 10)
ax[2].grid()

for i in range(0,3):
    ax[i].set_xlim([1,2e3])
plt.show()