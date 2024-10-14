import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt, chirp
from pywavelet.transforms import (
    from_time_to_wavelet,
    from_freq_to_wavelet,
    from_wavelet_to_time,
    from_wavelet_to_freq,
)
from pywavelet.transforms.types import FrequencySeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd

import os

OUTDIR = "out_filtered_chirp"
os.makedirs(OUTDIR, exist_ok=True)



def bandpass_data(rawstrain, f_min_bp, f_max_bp, srate_dt, bandpassing_flag):
    """
    Bandpass the raw strain between [f_min, f_max] Hz.
    """
    if bandpassing_flag:
        bb, ab = butter(2, [f_min_bp / (0.5 * srate_dt), f_max_bp / (0.5 * srate_dt)], btype='band')
        strain = filtfilt(bb, ab, rawstrain)
    else:
        strain = rawstrain
    return strain





def plot_time_and_frequency_domain(t, y, y_fft, freq, filtered_y, filtered_y_fft, out_dir):
    """Plot time and frequency domain of the signals."""
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    ax[0, 0].plot(t, y)
    ax[0, 0].set_xlabel('Time [seconds]')
    ax[0, 0].set_ylabel('Strain')
    ax[0, 0].set_title('Chirping Sinusoid')
    ax[0, 0].grid()

    ax[0, 1].loglog(freq, abs(y_fft) ** 2)
    ax[0, 1].set_xlabel('Frequency [Hz]')
    ax[0, 1].set_ylabel('Periodogram')
    ax[0, 1].set_title('Frequency Domain')
    ax[0, 1].grid()

    ax[1, 0].plot(t, filtered_y)
    ax[1, 0].set_xlabel('Time [seconds]')
    ax[1, 0].set_ylabel('Strain')
    ax[1, 0].set_title('Filtered Chirping Sinusoid')
    ax[1, 0].grid()

    ax[1, 1].loglog(freq, abs(filtered_y_fft) ** 2)
    ax[1, 1].set_xlabel('Frequency [Hz]')
    ax[1, 1].set_ylabel('Periodogram')
    ax[1, 1].set_title('Filtered Frequency Domain')
    ax[1, 1].grid()

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_and_freq_domain.png"))
    plt.clf()


def plot_comparison(t, y, filtered_y, freq, y_fft, filtered_y_fft, low_pass, high_pass, out_dir):
    """Plot comparison of the original and filtered signals in both time and frequency domain."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))

    ax[0].plot(t, y, label='Truth')
    ax[0].plot(t, filtered_y, label='Filtered', c='red', linestyle='--')
    ax[0].set_xlabel("Time [seconds]")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_title("Filtered Time Series")
    ax[0].legend()

    ax[1].loglog(freq, abs(y_fft) ** 2, label='Truth')
    ax[1].loglog(freq, abs(filtered_y_fft) ** 2, label='Filtered', linestyle='--', c='red')
    ax[1].axvline(low_pass, c='black', linestyle='--', label='Low Frequency Filter')
    ax[1].axvline(high_pass, c='black', linestyle='--', label='High Frequency Filter')
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].set_ylabel("Amplitude")
    ax[1].set_title("Filtered Frequency Series")
    ax[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison.png"))
    plt.clf()


def plot_wavelet_grid(signal_wavelet_freq, signal_wavelet_freq_filtered, out_dir):
    """Plot wavelet grid for both the original and filtered signals."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))

    signal_wavelet_freq.plot(ax=ax[0], zscale="linear", freq_scale="linear", absolute=False, freq_range=[0, 300])
    ax[0].set_xlabel('Wavelet Time Bins')
    ax[0].set_ylabel('Wavelet Frequency Bins')
    ax[0].set_title('Chirping Sinusoid')

    signal_wavelet_freq_filtered.plot(ax=ax[1], zscale="linear", freq_scale="linear", absolute=False,
                                      freq_range=[0, 300])
    ax[1].set_xlabel('Wavelet Time Bins')
    ax[1].set_ylabel('Wavelet Frequency Bins')
    ax[1].set_title('Filtered Chirping Sinusoid')

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "wavelet_grid.png"))
    plt.clf()


def main():

    print("Starting the process...")

    # Parameters
    f0 = 0
    f1 = 300
    T = 10
    A = 3.0
    PSD_AMP = 1e-2

    # Sampling parameters
    dt = 0.01 / (2 * f1)
    t = np.arange(0, T, dt)
    t = t[:2 ** int(np.log2(len(t)))]  # Round len(t) to the nearest power of 2
    T = len(t) * dt
    ND = len(t)

    Nf = 2 ** 13  # Number of wavelet frequency bins

    print("Generating chirp signal...")
    # Generate chirp signal
    y = A * chirp(t, f0, T, f1, method='quadratic', phi=0, vertex_zero=True)
    window = tukey(len(y), alpha=0.2)
    y *= window

    freq = np.fft.rfftfreq(len(y), dt)
    freq[0] = freq[1]
    y_fft = np.fft.rfft(y)

    print("Applying bandpass filter...")
    # Bandpass filter the signal
    low_pass, high_pass = 50, 300
    filtered_y = bandpass_data(y, low_pass, high_pass, 1 / dt, True)
    filtered_y_fft = np.fft.rfft(filtered_y)

    # Plot time and frequency domain
    plot_time_and_frequency_domain(t, y, y_fft, freq, filtered_y, filtered_y_fft, OUTDIR)

    # Comparison plot
    print("Creating comparison plots...")
    plot_comparison(t, y, filtered_y, freq, y_fft, filtered_y_fft, low_pass, high_pass, OUTDIR)

    # Power spectral density
    PSD = PSD_AMP * np.ones(len(freq))

    # Compute SNR in frequency domain
    print("Computing SNR in frequency domain...")
    SNR2_f = 4 * dt * np.sum(abs(y_fft) ** 2 / (ND * PSD))

    print("SNR in frequency domain:", SNR2_f ** 0.5)

    # Convert to wavelet domain
    signal_frequencyseries = FrequencySeries(y_fft, freq=freq)
    signal_frequencyseries_filtered = FrequencySeries(filtered_y_fft, freq=freq)

    print("Converting to wavelet domain...")
    signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf=Nf)
    signal_wavelet_freq_filtered = from_freq_to_wavelet(signal_frequencyseries_filtered, Nf=Nf)

    # Compute PSD in wavelet domain
    psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_freq.freq,
        t_grid=signal_wavelet_freq.time,
        dt=dt,
    )

    # Compute SNR in wavelet domain
    print("Computing SNR in wavelet domain...")
    snr2_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2
    snr2_freq_to_wavelet_filtered = compute_snr(signal_wavelet_freq_filtered, psd_wavelet_freq) ** 2

    print("SNR using wavelet domain, freq -> wavelet:", snr2_freq_to_wavelet ** 0.5)
    print("SNR using wavelet domain, freq filtered -> wavelet:", snr2_freq_to_wavelet_filtered ** 0.5)

    # Plot wavelet grid
    print("Creating wavelet grid plots...")
    plot_wavelet_grid(signal_wavelet_freq, signal_wavelet_freq_filtered, OUTDIR)

    print("Process complete!")


if __name__ == '__main__':
    main()
