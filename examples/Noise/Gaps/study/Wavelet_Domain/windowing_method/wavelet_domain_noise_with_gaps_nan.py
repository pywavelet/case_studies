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

from gap_study_utils.noise_curves import noise_PSD_AE
from gap_study_utils.signal_utils import waveform

# Constants
np.random.seed(1234)
ONE_HOUR = 60 * 60


def gap_routine_nan(t, start_window, end_window, delta_t=10):
    """
    Function to insert NaNs into a time series during a specific gap.
    """
    start_window *= ONE_HOUR  # Define gap_start of gap
    end_window *= ONE_HOUR  # Define gap_end of gap

    nan_window = [
        (np.nan if (start_window < time < end_window) else 1) for time in t
    ]

    return nan_window


def generate_waveform(a_true, f_true, fdot_true, t):
    """
    Generates the waveform, applies zero padding, and returns time-padded and frequency-domain waveforms.
    """
    h_t = waveform(a_true, f_true, fdot_true, t)
    taper_signal = tukey(len(h_t), alpha=0.0)
    h_t_pad = zero_pad(h_t * taper_signal)
    return h_t_pad


def compute_snr(h_f, PSD, delta_t, N):
    """
    Computes the matched filtering SNR for a given signal and noise PSD.
    """
    return np.sqrt(inner_prod(h_f, h_f, PSD, delta_t, N))


def wavelet_transform(signal_f_series, PSD, delta_t, Nf):
    """
    Computes the wavelet transform of the signal and the evolutionary PSD.
    """
    h_wavelet = from_freq_to_wavelet(signal_f_series, Nf=Nf)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=PSD.data,
        psd_f=PSD.freq,
        f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time,
        dt=delta_t,
    )
    return h_wavelet, psd_wavelet


def generate_noise_matrices(
    freq, variance_noise_f, w_t, delta_t, kwgs, num_samples=500
):
    """
    Generates noise matrices for both the regular and gap-inserted data.
    """
    noise_wavelet_matrices = []
    noise_gap_wavelet_matrices = []

    for i in tqdm(range(num_samples)):
        np.random.seed(i)
        noise_f_iter = np.random.normal(
            0, np.sqrt(variance_noise_f)
        ) + 1j * np.random.normal(0, np.sqrt(variance_noise_f))
        noise_f_iter[0] = np.sqrt(2) * noise_f_iter[0].real
        noise_f_iter[-1] = np.sqrt(2) * noise_f_iter[-1].real

        noise_f_freq_series = FrequencySeries(noise_f_iter, freq=freq)
        noise_wavelet = from_freq_to_wavelet(noise_f_freq_series, **kwgs)

        noise_t_iter = np.fft.irfft(
            noise_f_iter
        )  # Compute stationary noise in TD
        noise_t_gap_iter = w_t * noise_t_iter  # Apply gaps in TD noise
        noise_f_gap_iter = np.fft.rfft(noise_t_gap_iter)

        noise_f_freq_gap_series = FrequencySeries(noise_f_gap_iter, freq=freq)
        noise_gap_wavelet = from_freq_to_wavelet(
            noise_f_freq_gap_series, **kwgs
        )

        noise_wavelet_matrices.append(noise_wavelet.data)
        noise_gap_wavelet_matrices.append(noise_gap_wavelet.data)

    return np.array(noise_wavelet_matrices), np.array(
        noise_gap_wavelet_matrices
    )


def compute_covariance_matrices(
    noise_wavelet_matrix, noise_gap_wavelet_matrix
):
    """
    Computes the covariance matrices for the regular and gap-inserted noise data.
    """
    N_f, N_t = noise_wavelet_matrix.shape[1], noise_wavelet_matrix.shape[2]
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


def plot_wavelets(
    h_wavelet,
    cov_matrix_wavelet,
    h_wavelet_gap,
    cov_matrix_gap_wavelet,
    psd_wavelet,
    noise_gap_wavelet,
    N_f,
    N_t,
    start_window,
    end_window,
    lobe_length,
    tmax,
):
    """
    Plots the wavelet spectrograms and saves the figure.
    """
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

    breakpoint()
    plot_wavelet_grid(
        cov_matrix_gap_wavelet,
        time_grid=noise_gap_wavelet.time / 60 / 60,
        freq_grid=noise_gap_wavelet.freq,
        ax=ax[1, 0],
        zscale="linear",
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
        f"wavelet_Nf_{N_f}_Nt_{N_t}_start_{start_window}_end_{end_window}_lobe_length_{lobe_length}_tmax_{tmax}.pdf",
        bbox_inches="tight",
    )
    plt.show()
    plt.clf()


def main(a_true=1e-21, f_true=3e-3, fdot_true=1e-8, TDI="TDI1"):
    start_window, end_window, lobe_length = 4, 6, 1
    Nf = 128
    tmax = 10 * ONE_HOUR  # 10 hours
    fs = 2 * f_true  # Sampling rate
    delta_t = np.floor(0.4 / fs)
    t = np.arange(0, tmax, delta_t)
    N = int(2 ** (np.ceil(np.log2(len(t)))))

    # Generate waveform and padded data
    h_t_pad = generate_waveform(a_true, f_true, fdot_true, t)
    t_pad = np.arange(0, len(h_t_pad) * delta_t, delta_t)

    h_true_f = np.fft.rfft(h_t_pad)
    freq = np.fft.rfftfreq(N, delta_t)
    freq[0] = freq[1]
    PSD = noise_PSD_AE(freq, TDI=TDI)

    # Compute SNR in time and wavelet domain
    SNR2 = compute_snr(h_true_f, PSD, delta_t, N)
    print("SNR of source:", SNR2)

    signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
    psd = FrequencySeries(data=PSD, freq=freq)

    h_wavelet, psd_wavelet = wavelet_transform(
        signal_f_series, psd, delta_t, Nf
    )
    SNR2_wavelet = np.nansum(((h_wavelet * h_wavelet) / psd_wavelet).data)
    print("SNR in wavelet domain:", np.sqrt(SNR2_wavelet))

    variance_noise_f = N * PSD / (4 * delta_t)

    # Generate gaps and noise matrices
    w_t = gap_routine_nan(
        t_pad,
        start_window=start_window,
        end_window=end_window,
        delta_t=delta_t,
    )
    noise_wavelet_matrix, noise_gap_wavelet_matrix = generate_noise_matrices(
        freq, variance_noise_f, w_t, delta_t, kwgs={"Nf": Nf}
    )

    # Covariance matrices
    cov_matrix_wavelet, cov_matrix_gap_wavelet = compute_covariance_matrices(
        noise_wavelet_matrix, noise_gap_wavelet_matrix
    )
    breakpoint()

    # Apply gaps to signal and retransform to wavelet domain
    h_t_gap = h_t_pad * w_t
    h_f_gap = np.fft.rfft(h_t_gap)

    h_wavelet_gap, psd_gap_wavelet = wavelet_transform(
        FrequencySeries(data=h_f_gap, freq=freq), psd, delta_t, Nf
    )
    SNR_gap_wavelet = np.nansum(
        ((h_wavelet_gap * h_wavelet_gap) / psd_gap_wavelet).data
    )
    print("SNR with gaps in wavelet domain:", np.sqrt(SNR_gap_wavelet))

    # Plot
    breakpoint()
    plot_wavelets(
        h_wavelet,
        cov_matrix_wavelet,
        h_wavelet_gap,
        cov_matrix_gap_wavelet,
        psd_wavelet,
        h_wavelet_gap,
        Nf,
        N,
        start_window,
        end_window,
        lobe_length,
        tmax,
    )


if __name__ == "__main__":
    main()
