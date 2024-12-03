import os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import normal
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries
from pywavelet.transforms.types.plotting import plot_wavelet_grid
from pywavelet.utils import evolutionary_psd_from_stationary_psd
from scipy.signal.windows import tukey
from tqdm import tqdm

from gap_study_utils.noise_curves import noise_PSD_AE
from gap_study_utils.signal_utils import waveform


# Function to generate waveform
def generate_waveform(a_true, f_true, fdot_true, t, delta_t):
    h_t = waveform(a_true, f_true, fdot_true, t)
    taper_signal = tukey(len(h_t), alpha=0.2)
    h_t_pad = zero_pad(h_t * taper_signal)
    t_pad = np.arange(0, len(h_t_pad) * delta_t, delta_t)
    return h_t_pad, t_pad


# Function to compute SNR in frequency domain
def compute_snr(h_true_f, PSD, delta_t, N):
    return np.sqrt(inner_prod(h_true_f, h_true_f, PSD, delta_t, N))


# Function to compute evolutionary PSD in wavelet domain
def compute_evolutionary_psd(signal_f_series, psd, Nf, delta_t):
    h_wavelet = from_freq_to_wavelet(signal_f_series, Nf=Nf)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time,
        dt=delta_t,
    )
    return h_wavelet, psd_wavelet


# Function to compute wavelet SNR
def compute_wavelet_snr(h_wavelet, psd_wavelet):
    SNR2_wavelet = np.nansum(((h_wavelet * h_wavelet) / psd_wavelet).data)
    return np.sqrt(SNR2_wavelet)


# Function to estimate covariance matrix in wavelet domain
def estimate_covariance_matrix(noise_wavelet_matrix, N_f, N_t):
    cov_matrix_wavelet = np.zeros((N_f, N_t), dtype=float)
    for i in range(N_f):
        for j in range(N_t):
            cov_matrix_wavelet[i, j] = np.cov(
                noise_wavelet_matrix[:, i, j], rowvar=False
            )
    return cov_matrix_wavelet


# Function to generate noise in frequency domain and convert to wavelet domain
def generate_noise_wavelet(freq, PSD, Nf, N, delta_t, num_iterations=20000):
    variance_noise_f = N * PSD / (4 * delta_t)
    noise_wavelet_matrices = []
    for i in tqdm(range(num_iterations)):
        np.random.seed(i)
        noise_f_iter = np.random.normal(
            0, np.sqrt(variance_noise_f)
        ) + 1j * np.random.normal(0, np.sqrt(variance_noise_f))
        noise_f_iter[0] = np.sqrt(2) * noise_f_iter[0].real
        noise_f_iter[-1] = np.sqrt(2) * noise_f_iter[-1].real

        noise_f_freq_series = FrequencySeries(noise_f_iter, freq=freq)
        noise_wavelet = from_freq_to_wavelet(noise_f_freq_series, Nf=Nf)
        noise_wavelet_matrices.append(noise_wavelet.data)

    return np.array(noise_wavelet_matrices)


# Main function to run the pipeline
def main(
    a_true=1e-21, f_true=3e-3, fdot_true=1e-8, TDI="TDI1", tmax=10 * 60 * 60
):
    fs = 2 * f_true
    delta_t = np.floor(0.4 / fs)

    t = np.arange(0, tmax, delta_t)
    N = int(2 ** (np.ceil(np.log2(len(t)))))

    # Generate waveform
    h_t_pad, t_pad = generate_waveform(a_true, f_true, fdot_true, t, delta_t)

    # Frequency domain analysis
    h_true_f = np.fft.rfft(h_t_pad)
    freq = np.fft.rfftfreq(N, delta_t)
    freq[0] = freq[1]
    PSD = noise_PSD_AE(freq, TDI=TDI)

    # Compute SNR in frequency domain
    SNR_freq = compute_snr(h_true_f, PSD, delta_t, N)
    print("SNR in frequency domain:", SNR_freq)

    # Wavelet domain analysis
    signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
    psd = FrequencySeries(data=PSD, freq=freq)

    Nf = 32
    h_wavelet, psd_wavelet = compute_evolutionary_psd(
        signal_f_series, psd, Nf, delta_t
    )

    SNR_wavelet = compute_wavelet_snr(h_wavelet, psd_wavelet)
    print("SNR in wavelet domain:", SNR_wavelet)

    # Generate noise and estimate covariance matrix in wavelet domain
    noise_wavelet_matrix = generate_noise_wavelet(freq, PSD, Nf, N, delta_t)
    cov_matrix_wavelet = estimate_covariance_matrix(
        noise_wavelet_matrix, h_wavelet.data.shape[0], h_wavelet.data.shape[1]
    )

    # Plot results
    fig, ax = plt.subplots(2, 1)
    plot_wavelet_grid(
        cov_matrix_wavelet,
        time_grid=h_wavelet.time,
        freq_grid=h_wavelet.freq,
        ax=ax[0],
        zscale="log",
        freq_scale="linear",
        absolute=True,
        freq_range=[h_wavelet.freq[1], 5e-3],
    )

    plot_wavelet_grid(
        psd_wavelet.data,
        time_grid=psd_wavelet.time,
        freq_grid=psd_wavelet.freq,
        ax=ax[1],
        zscale="log",
        freq_scale="linear",
        absolute=True,
        freq_range=[psd_wavelet.freq[1], 5e-3],
    )

    plt.savefig("cov_matrix_wavelet.pdf", bbox_inches="tight")

    # Compute estimated SNR in wavelet domain using covariance matrix
    SNR_estimated_wavelet = np.sqrt(
        np.nansum((h_wavelet * h_wavelet) / cov_matrix_wavelet)
    )
    print("Estimated SNR using wavelet covariance:", SNR_estimated_wavelet)


# Run the pipeline
if __name__ == "__main__":
    main()
