import os

import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, sosfiltfilt, sos2zpk, sosfreqz
from numpy.random import normal

from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import FrequencySeries
from pywavelet.utils.lisa import waveform, zero_pad
import sys
sys.path.append("../")
from noise_curves import noise_PSD_AE, CornishPowerSpectralDensity
from gap_funcs import gap_routine, get_frequency_response, bandpass_data, get_Cov
np.random.seed(1234)


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

# Set true parameters. These are the parameters we want to estimate using MCMC.

a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

TDI = "TDI1"
start_window = 4
end_window = 6
lobe_length = 1

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
delta_f = freq[2] - freq[1] 

if TDI == "TDI1" or TDI == "TDI2":
    PSD = noise_PSD_AE(freq, TDI = TDI)
else:
    PSD = CornishPowerSpectralDensity(freq)
SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))



# Compute things in the wavelet domain

signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
psd = FrequencySeries(data = PSD, freq = freq)


kwgs = dict(
    Nf=512,
)


h_wavelet = Data.from_frequencyseries(signal_f_series, **kwgs).wavelet
psd_wavelet = evolutionary_psd_from_stationary_psd(
                                                    psd=psd.data,
                                                    psd_f=psd.freq,
                                                    f_grid=h_wavelet.freq,
                                                    t_grid=h_wavelet.time,
                                                    dt=delta_t,
                                                )

SNR2_wavelet = np.nansum((h_wavelet*h_wavelet) / psd_wavelet)
print("SNR in wavelet domain is", SNR2_wavelet**(1/2))

# Gaps in the frequency domain. Now to introduce a filter.  
f_low_index = np.argwhere(freq >= 1e-4)[0][0]
order_filter = 2
w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)

## ================ FILTERING STEP =================

filtered_w_t = bandpass_data(w_t, freq[f_low_index], 1/delta_t, bandpassing_flag = True, order = order_filter) 
# Check
freqs_from_filter, response_f = get_frequency_response(freq[f_low_index], 1/delta_t, N,  order=order_filter)

filter_response_function_double = abs(response_f)**2
# breakpoint()

check_filter = abs(response_f)**2 * np.fft.rfft(w_t)

filtered_w_fft = np.fft.rfft(filtered_w_t)

# import matplotlib.pyplot as plt
# plt.loglog(freqs_from_filter, abs(check_filter)**2, '*', label = 'K(f) * W(f)')
# plt.loglog(freqs_from_filter, abs(filtered_w_fft)**2, '*', label = 'filtered')
# plt.legend()
# plt.show()
# breakpoint()
# quit()

# filtered_w_t = w_t
# f_low_index = 0
freq_bins_pos_neg = np.fft.fftshift(np.fft.fftfreq(len(filtered_w_t), delta_t))
freq_bins_pos_neg[N//2] = freq_bins_pos_neg[N//2 + 1]

if TDI == "TDI1" or TDI == "TDI2":
    PSD_pos_neg = noise_PSD_AE(freq_bins_pos_neg, TDI = TDI)
else:
    PSD_pos_neg = CornishPowerSpectralDensity(freq_bins_pos_neg)

delta_f = freq[2] - freq[1]

# - use positive and negative frequencies. 
w_fft = np.fft.fftshift(np.fft.fft(w_t))  # Compute fft of windowing function (neg_pos_freq)
w_star_fft = np.conjugate(w_fft)  # Compute conjugate 

Cov_Matrix = np.zeros(shape=(N//2 + 1,N//2 + 1), dtype=complex) # Matrix will be filled full of complex numbers.
                                                               # here we only have positive frequencies

# Build analytical covariance matrix
print("Building the analytical covariance matrix")
Cov_Matrix_Gated = get_Cov(Cov_Matrix, w_fft, w_star_fft, delta_f, PSD_pos_neg)

filter_response_vec = filter_response_function_double.reshape(-1,1)
filter_matrix = filter_response_vec @ filter_response_vec.T

Cov_Matrix_Gated = filter_matrix * Cov_Matrix
breakpoint()
os.chdir('../Data/')
if TDI == "TDI1":
    np.save("Cov_Matrix_analytical_gap_TDI1_filtered.npy", Cov_Matrix_Gated)
    # np.save("Cov_Matrix_analytical_inv_regularised_TDI1_filtered.npy", Cov_Matrix_Gated_Inv_Regularised)
elif TDI == "TDI2":
    np.save("Cov_Matrix_analytical_gap_TDI2_filtered.npy", Cov_Matrix_Gated)
    # np.save("Cov_Matrix_analytical_inv_regularised_TDI2_filtered.npy", Cov_Matrix_Gated_Inv_Regularised)
else:
    np.save("Cov_Matrix_analytical_gap_Cornish_filtered.npy", Cov_Matrix_Gated)
    # np.save("Cov_Matrix_analytical_inv_regularised_Cornish_filtered.npy", Cov_Matrix_Gated_Inv_Regularised)

quit()
Cov_Matrix_filtered = Cov_Matrix_Gated.copy()
Cov_Matrix_filtered = Cov_Matrix_filtered[f_low_index:, f_low_index:]

breakpoint()

Cov_Matrix_filtered_Gated_Inv = np.linalg.inv(Cov_Matrix_filtered)


# Regularise covariance matrix (stop it being singular)
# zero_points_window = np.argwhere(w_t == 0)[1][0]
# tol = w_t[zero_points_window - 1] # Last moment before w_t is nonzero
# Cov_Matrix_Gated_Inv_Regularised = regularise_matrix(Cov_Matrix_Gated, w_t, tol = tol)

Cov_Matrix_Stat = np.linalg.inv(np.diag(N*PSD/(2*delta_t)))

# =================== Compute SNRs ====================================
h_w_t = h_t_pad * w_t
h_w_fft = np.fft.rfft(h_w_t)

filtered_h_w = bandpass_data(h_w_t, freq[f_low_index], 1/delta_t, bandpassing_flag = True, order = 4) 
filtered_h_w_fft = np.fft.rfft(filtered_h_w)[f_low_index:]

SNR2_gaps = np.real((2*filtered_h_w_fft.conj() @ Cov_Matrix_filtered_Gated_Inv @ filtered_h_w_fft))
SNR2_gaps_regularised = np.real((2*h_w_fft.conj() @ Cov_Matrix_Gated_Inv_Regularised @ h_w_fft))
SNR2_no_gaps = np.real((2*h_true_f.conj() @ Cov_Matrix_Stat @ h_true_f))

print("SNR when there are gaps in the frequency domain:", SNR2_gaps**(1/2))
print("SNR when there are gaps in the frequency domain using regularised matrix:", SNR2_gaps_regularised**(1/2))
print("SNR when there are no gaps in the frequency domain:", SNR2_no_gaps**(1/2))







