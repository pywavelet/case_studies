import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from scipy.signal.windows import tukey
from pywavelet.utils.lisa import waveform, zero_pad, inner_prod
sys.path.append("../")
from noise_curves import noise_PSD_AE, CornishPowerSpectralDensity
from gap_funcs import gap_routine, regularise_matrix, bandpass_data 
np.random.seed(1234)

a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

# TDI = "Cornish" # TDI1 red noise, TDI2 white noise at low f 
TDI = "TDI1" # TDI1 red noise, TDI2 white noise at low f 

# Windowing parameters 
start_window = 4
end_window = 6
lobe_length = 1

tmax = 10 * 60 * 60  # Final time
fs = 2 * f_true  # Sampling rate
delta_t = np.floor(0.4 / fs) # Sampling interval

t = np.arange(0, tmax, delta_t)  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

# Round length of time series to a power of two.
N = int(2 ** (np.ceil(np.log2(len(t)))))  

# Gen waveform, taper and then pad 
h_t = waveform(a_true, f_true, fdot_true, t)
taper_signal = tukey(len(h_t), alpha = 0.2)
h_t_pad = zero_pad(h_t * taper_signal)

# Compute new time array corresponding to padded signal
t_pad = np.arange(0,len(h_t_pad)*delta_t, delta_t)

# Fourier domain, gen frequencies, PSD and signal in freq
freq = np.fft.rfftfreq(N, delta_t); #freq[0] = freq[1] # set f_0 = f_1 for PSD
if TDI == 'TDI1' or TDI == 'TDI2':
    PSD = noise_PSD_AE(freq, TDI = TDI)
elif TDI == "Cornish":
    PSD = CornishPowerSpectralDensity(freq)

PSD[0] = PSD[1]

h_true_f = np.fft.rfft(h_t_pad)

# Compute SNR
SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  
print("SNR of source", np.sqrt(SNR2))

# Gaps in the frequency domain. 

# Generate gaps and apply them to signal
w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)
h_w_pad = w_t * h_t_pad
h_w_fft = np.fft.rfft(h_w_pad)
variance_noise_f = N * PSD/(4*delta_t)   # Compute variance in frequency domain (pos freq)

# Generate covariance of noise for stat process 
cov_matrix_stat = np.linalg.inv(np.diag(2*variance_noise_f))

# ====================== ESTIMATE THE NOISE COVARIANCE MATRIX ==============================
print("Estimating the gated covariance matrix")

f_low_index = np.argwhere(freq >= 1e-4)[0][0]
noise_f_gap_filtered_vec = []
# w_t = bandpass_data(w_t, freq[f_low_index], 1/delta_t, bandpassing_flag = True, order = 4) 
for i in tqdm(range(0,80000)):
    np.random.seed(i)
    noise_f_iter = np.random.normal(0,np.sqrt(variance_noise_f))  + 1j * np.random.normal(0,np.sqrt(variance_noise_f)) 
    noise_f_iter[0] = np.sqrt(2)*noise_f_iter[0].real
    noise_f_iter[-1] = np.sqrt(2)*noise_f_iter[-1].real

    noise_t_iter = np.fft.irfft(noise_f_iter)      # Compute stationary noise in TD
    noise_t_gap_iter = w_t * noise_t_iter  # Place gaps in the noise from the TD

    noise_t_gap_filtered = bandpass_data(noise_t_gap_iter, freq[f_low_index], 1/delta_t, bandpassing_flag = True, order = 4)
    time = np.arange(0, delta_t * len(noise_t_gap_filtered), delta_t)

    # plt.plot(time/60/60, noise_t_gap_iter, c = 'blue');
    # plt.plot(time/60/60, noise_t_gap_filtered, c = 'red');
    # plt.xlabel(r'Time') 
    # plt.show()
    # breakpoint()
    noise_f_gap_filtered_iter = np.fft.rfft(noise_t_gap_filtered) # Convert into FD
    noise_f_gap_filtered_vec.append(noise_f_gap_filtered_iter) 

# ==========================================================================================

print("Now estimating covariance matrix")    
cov_matrix_freq_gap = np.cov(noise_f_gap_filtered_vec,rowvar = False)

os.chdir('../Data/')
if TDI == "TDI1":
    np.save("Cov_Matrix_estm_gap_TDI1_filtered.npy", cov_matrix_freq_gap)
    # np.save("Cov_Matrix_analytical_inv_regularised_TDI1_filtered.npy", Cov_Matrix_Gated_Inv_Regularised)
elif TDI == "TDI2":
    np.save("Cov_Matrix_estm_gap_TDI2_filtered.npy", cov_matrix_freq_gap)
    # np.save("Cov_Matrix_analytical_inv_regularised_TDI2_filtered.npy", Cov_Matrix_Gated_Inv_Regularised)
else:
    np.save("Cov_Matrix_estm_gap_Cornish_filtered.npy", cov_matrix_freq_gap)
    # np.save("Cov_Matrix_analytical_inv_regularised_Cornish_filtered.npy", Cov_Matrix_Gated_Inv_Regularised)

quit()
breakpoint()
# Regularise covariance matrix (stop it being singular)
zero_points_window = np.argwhere(w_t == 0)[1][0]
tol = w_t[zero_points_window - 1] # Last moment before w_t is nonzero
cov_matrix_freq_gap_regularised_inv = regularise_matrix(cov_matrix_freq_gap, w_t, tol = tol)
print("Finished estimating the covariance matrix")

# cov matrix freq gap
cov_matrix_freq_gap_inv = np.linalg.inv(cov_matrix_freq_gap) # Compute inverse of estimated matrix

# Compute various SNRs
SNR2_gaps = np.real((2*h_w_fft.conj() @ cov_matrix_freq_gap_inv @ h_w_fft))
SNR2_gaps_regularised = np.real((2*h_w_fft.conj() @ cov_matrix_freq_gap_regularised_inv @ h_w_fft))
SNR2_no_gaps = np.real((2*h_true_f.conj() @ cov_matrix_stat @ h_true_f))

print("SNR when there are gaps in the frequency domain", SNR2_gaps**0.5)
print("SNR when there are gaps in the frequency domain is", SNR2_gaps_regularised**0.5,"using regularised matrix")
print("SNR when there are no gaps in the frequency domain", SNR2_no_gaps**0.5)

# Save the data


np.save("../Data/Cov_Matrix_estm_gap.npy", cov_matrix_freq_gap)
np.save("../Data/Cov_Matrix_estm_inv_regularised.npy", cov_matrix_freq_gap_regularised_inv)

