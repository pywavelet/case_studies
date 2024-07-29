import os


import numpy as np
from tqdm import tqdm

from numpy.random import normal

from pywavelet.utils.lisa import waveform, zero_pad, inner_prod
from noise_curves import noise_PSD_AE

from gap_funcs import gap_routine 
np.random.seed(1234)

a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

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
h_t_pad = zero_pad(h_t)

t_pad = np.arange(0,len(h_t_pad)*delta_t, delta_t)

freq = np.fft.rfftfreq(N, delta_t); freq[0] = freq[1]
PSD = noise_PSD_AE(freq, TDI = "TDI1")

breakpoint()
h_true_f = np.fft.rfft(h_t_pad)

SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))

# Gaps in the frequency domain. 
w_t = gap_routine(t_pad, start_window = 4, end_window = 6, lobe_length = 1, delta_t = delta_t)

h_w_pad = w_t * h_t_pad

h_w_fft = np.fft.rfft(h_w_pad)
variance_noise_f = N * PSD/(4*delta_t)   # Compute variance in frequency domain (pos freq)

cov_matrix_stat = np.linalg.inv(np.diag(2*variance_noise_f))
# ====================== ESTIMATE THE NOISE COVARIANCE MATRIX ==============================
print("Estimating the gated covariance matrix")
noise_f_gap_vec = []
for i in tqdm(range(0,100000)):
    np.random.seed(i)
    noise_f_iter = np.random.normal(0,np.sqrt(variance_noise_f))  + 1j * np.random.normal(0,np.sqrt(variance_noise_f)) 
    noise_f_iter[0] = np.sqrt(2)*noise_f_iter[0].real
    noise_f_iter[-1] = np.sqrt(2)*noise_f_iter[-1].real

    noise_t_iter = np.fft.irfft(noise_f_iter)      # Compute stationary noise in TD
    noise_t_gap_iter = w_t * noise_t_iter  # Place gaps in the noise from the TD
    noise_f_gap_iter = np.fft.rfft(noise_t_gap_iter) # Convert into FD
    noise_f_gap_vec.append(noise_f_gap_iter) 

# ==========================================================================================

print("Now estimating covariance matrix")    
cov_matrix_freq_gap = np.cov(noise_f_gap_vec,rowvar = False)
print("Finished estimating the covariance matrix")

cov_matrix_freq_gap_inv = np.linalg.inv(cov_matrix_freq_gap)

breakpoint()

SNR2_gaps = np.real((2*h_w_fft.conj() @ cov_matrix_freq_gap_inv @ h_w_fft))
SNR2_no_gaps = np.real((2*h_true_f.conj() @ cov_matrix_stat @ h_true_f))

print("SNR when there are gaps in the frequency domain", SNR2_gaps**(1/2))
print("SNR when there are no gaps in the frequency domain", SNR2_gaps**(1/2))


os.chdir('Data/')
np.save("Cov_Matrix_estm_gap.npy", cov_matrix_freq_gap)
