import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from pywavelet.transforms import (
    from_freq_to_wavelet,
    from_time_to_wavelet,
    from_wavelet_to_time,
)
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.transforms.types.plotting import plot_wavelet_grid
from pywavelet.utils import evolutionary_psd_from_stationary_psd
from scipy.signal.windows import tukey
from tqdm import tqdm

from gap_study_utils.gap_funcs import gap_routine
from gap_study_utils.noise_curves import noise_PSD_AE
from gap_study_utils.signal_utils import waveform


def generate_time_series(a_true, f_true, fdot_true, tmax, delta_t):
    t = np.arange(0, tmax, delta_t)
    N = int(2 ** (np.ceil(np.log2(len(t)))))
    h_t = waveform(a_true, f_true, fdot_true, t)
    return t, h_t, N


def apply_tukey_window(signal):
    return tukey(len(signal), alpha=0.2) * signal


def compute_frequency_domain(h_t_pad, delta_t, N):
    h_true_f = np.fft.rfft(h_t_pad)
    freq = np.fft.rfftfreq(N, delta_t)
    freq[0] = freq[1]  # Avoid division by zero
    return h_true_f, freq


def compute_psd(freq, TDI="TDI1"):
    return noise_PSD_AE(freq, TDI=TDI)


def compute_snr(h_f, PSD, delta_t, N):
    SNR2 = inner_prod(h_f, h_f, PSD, delta_t, N)
    return np.sqrt(SNR2)


def wavelet_transform(signal_f_series, psd, delta_t, Nf):
    kwgs = dict(Nf=Nf)
    h_wavelet = from_freq_to_wavelet(signal_f_series, **kwgs)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time,
        dt=delta_t,
    )
    return h_wavelet, psd_wavelet


def generate_gaps(t_pad, start_window, end_window, lobe_length, delta_t):
    return gap_routine(
        t_pad,
        start_window=start_window,
        end_window=end_window,
        lobe_length=lobe_length,
        delta_t=delta_t,
    )


def simulate_noise(variance_noise_f, freq, w_t, delta_t, N, Nf, n_iter=500):
    noise_wavelet_matrices, noise_gap_wavelet_matrices = [], []
    kwgs = dict(Nf=Nf)

    for i in tqdm(range(n_iter)):
        np.random.seed(i)
        noise_f_iter = np.random.normal(
            0, np.sqrt(variance_noise_f)
        ) + 1j * np.random.normal(0, np.sqrt(variance_noise_f))
        noise_f_iter[0] = np.sqrt(2) * noise_f_iter[0].real
        noise_f_iter[-1] = np.sqrt(2) * noise_f_iter[-1].real

        noise_f_freq_series = FrequencySeries(noise_f_iter, freq=freq)
        noise_wavelet = from_freq_to_wavelet(noise_f_freq_series, **kwgs)

        noise_t_iter = np.fft.irfft(noise_f_iter)
        noise_t_gap_iter = w_t * noise_t_iter
        noise_f_gap_iter = np.fft.rfft(noise_t_gap_iter)

        noise_f_freq_gap_series = FrequencySeries(noise_f_gap_iter, freq=freq)
        noise_gap_wavelet = from_freq_to_wavelet(
            noise_f_freq_gap_series, **kwgs
        )

        noise_wavelet_matrices.append(noise_wavelet.data)
        noise_gap_wavelet_matrices.append(noise_gap_wavelet.data)

    noise_wavelet_matrix = np.array(noise_wavelet_matrices)
    noise_gap_wavelet_matrix = np.array(noise_gap_wavelet_matrices)

    return noise_wavelet_matrix, noise_gap_wavelet_matrix


def compute_covariance_matrix(noise_wavelet_matrix, noise_gap_wavelet_matrix):
    N_f, N_t = noise_wavelet_matrix[0].shape

    cov_matrix_wavelet = np.zeros((N_f, N_t), dtype=float)
    cov_matrix_gap_wavelet = np.zeros((N_f, N_t), dtype=float)

    for i in range(N_f):
        for j in range(N_t):
            cov_matrix_wavelet[i, j] = np.cov(
                noise_wavelet_matrix[:, i, j], rowvar=False
            )
            cov_matrix_gap_wavelet[i, j] = np.cov(
                noise_gap_wavelet_matrix[:, i, j], rowvar=False
            )

    return cov_matrix_wavelet, cov_matrix_gap_wavelet


def plot_wavelet_spectrograms(
    h_wavelet,
    cov_matrix_wavelet,
    h_wavelet_gap,
    cov_matrix_gap_wavelet,
    psd_wavelet,
    Nf,
    N,
    start_window,
    end_window,
    lobe_length,
    tmax,
):
    fig, ax = plt.subplots(2, 2, figsize=(16, 8))

    freq_range = (0, 0.007)
    plot_wavelet_grid(
        cov_matrix_wavelet,
        time_grid=psd_wavelet.time / 60 / 60,
        freq_grid=psd_wavelet.freq,
        ax=ax[0, 0],
        zscale="log",
        freq_scale="linear",
        absolute=False,
        freq_range=freq_range,
    )
    ax[0, 0].set_title("Estimated wavelet covariance matrix")

    plot_wavelet_grid(
        h_wavelet.data,
        time_grid=h_wavelet.time / 60 / 60,
        freq_grid=h_wavelet.freq,
        ax=ax[0, 1],
        zscale="linear",
        freq_scale="linear",
        absolute=False,
        freq_range=freq_range,
    )
    ax[0, 1].set_title("Signal wavelet matrix")

    plot_wavelet_grid(
        cov_matrix_gap_wavelet,
        time_grid=h_wavelet_gap.time / 60 / 60,
        freq_grid=h_wavelet_gap.freq,
        ax=ax[1, 0],
        zscale="log",
        freq_scale="linear",
        absolute=False,
        freq_range=freq_range,
    )
    ax[1, 0].set_title("Wavelet covariance matrix with gaps in data")

    plot_wavelet_grid(
        h_wavelet_gap.data,
        time_grid=h_wavelet_gap.time / 60 / 60,
        freq_grid=h_wavelet_gap.freq,
        ax=ax[1, 1],
        zscale="linear",
        freq_scale="linear",
        absolute=False,
        freq_range=freq_range,
    )
    ax[1, 1].set_title("Signal wavelet matrix with gaps")

    plt.savefig(
        f"Spectrogram_Nf_{Nf}_Nt_{N//Nf}_start_{start_window}_end_{end_window}_lobe_length_{lobe_length}_tmax_{tmax}.pdf",
        bbox_inches="tight",
    )
    plt.show()


def main():
    a_true = 1e-21
    f_true = 3e-3
    fdot_true = 1e-8
    TDI = "TDI1"
    Nf = 128
    tmax = 10 * 60 * 60
    fs = 2 * f_true
    delta_t = np.floor(0.4 / fs)

    # Generate time series and apply window
    t, h_t, N = generate_time_series(a_true, f_true, fdot_true, tmax, delta_t)
    h_t_pad = zero_pad(apply_tukey_window(h_t))

    # Frequency domain
    h_true_f, freq = compute_frequency_domain(h_t_pad, delta_t, N)
    PSD = compute_psd(freq, TDI)

    # Compute SNR
    SNR = compute_snr(h_true_f, PSD, delta_t, N)
    print(f"SNR of source: {SNR}")

    # Wavelet transform and PSD
    signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
    psd = FrequencySeries(data=PSD, freq=freq)
    h_wavelet, psd_wavelet = wavelet_transform(
        signal_f_series, psd, delta_t, Nf
    )

    # Apply gaps
    w_t = generate_gaps(
        np.arange(0, len(h_t_pad) * delta_t, delta_t), 4, 6, 1, delta_t
    )
    h_pad_w = w_t * h_t_pad

    # Noise simulation
    variance_noise_f = N * PSD / (4 * delta_t)
    noise_wavelet_matrix, noise_gap_wavelet_matrix = simulate_noise(
        variance_noise_f, freq, w_t, delta_t, N, Nf
    )

    # Covariance matrix computation
    cov_matrix_wavelet, cov_matrix_gap_wavelet = compute_covariance_matrix(
        noise_wavelet_matrix, noise_gap_wavelet_matrix
    )

    # Plotting
    plot_wavelet_spectrograms(
        h_wavelet,
        cov_matrix_wavelet,
        h_wavelet,
        cov_matrix_gap_wavelet,
        psd_wavelet,
        Nf,
        N,
        4,
        6,
        1,
        tmax,
    )


if __name__ == "__main__":
    main()
