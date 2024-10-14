import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from scipy.signal import chirp

from pywavelet.transforms import (
    from_wavelet_to_time,
    from_wavelet_to_freq,
    from_time_to_wavelet, from_freq_to_wavelet
)
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd
import os

OUTDIR = "out_chirping_signal"
os.makedirs(OUTDIR, exist_ok=True)

# Function to generate chirp signal and apply window
def generate_chirp(f0, f1, T, A, dt, alpha):
    t = np.arange(0, T, dt)
    t = t[:2 ** int(np.log2(len(t)))]  # Round len(t) to nearest power of 2
    y = A * chirp(t, f0, T, f1, method='quadratic', phi=0, vertex_zero=True)
    window = tukey(len(y), alpha)  # Apply Tukey window
    y *= window
    return y, t

# Function to plot time and frequency domain signals
def plot_time_freq_domain(t, y, freq, y_fft, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5))
    ax[0].plot(t, y)
    ax[0].set_xlabel('Time [seconds]')
    ax[0].set_ylabel('Strain')
    ax[0].set_title('Chirping Sinusoid')
    ax[0].grid()

    ax[1].loglog(freq, abs(y_fft)**2)
    ax[1].set_xlabel('Frequency [Hz]')
    ax[1].set_ylabel('Periodigram')
    ax[1].set_title('Frequency domain')
    ax[1].grid()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()

# Function to plot wavelet transform
def plot_wavelet(signal_wavelet_freq, output_path, freq_range):
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    signal_wavelet_freq.plot(ax=ax, zscale="linear", freq_scale="linear", absolute=False, freq_range=freq_range)
    plt.savefig(output_path)
    plt.clf()

# Function to plot signal reconstructions
def plot_signal_reconstructions(t, y, signal_time_from_wavelet, output_path):
    fig, ax = plt.subplots(2, 2, figsize=(16, 7))
    for i in range(0, 2):
        for j in range(0, 2):
            ax[i, j].plot(t, y, label='truth', c='blue')
            ax[i, j].plot(t, signal_time_from_wavelet.data, c='red', linestyle='--', label='approx')
            ax[i, j].set_xlabel('Time [seconds]')
            ax[i, j].set_ylabel('Amplitude')
            ax[i, j].set_title('Comparing reconstructions')
            ax[i, j].legend()
            ax[i, j].grid()

    ax[0, 0].set_xlim([0, 1.5])
    ax[0, 1].set_xlim([1.9, 2.5])
    ax[1, 0].set_xlim([5, 5.05])
    ax[1, 1].set_xlim([5.43, 5.46])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()

# Function to plot residuals
def plot_residuals(freq, res_freq, t, res_time, output_path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    ax[0].loglog(freq, res_freq)
    ax[0].set_xlabel('Frequency')
    ax[0].set_ylabel('Residuals (squared)')
    ax[0].set_title('Frequency domain')
    ax[1].plot(t, res_time)
    ax[1].set_xlabel('Time [seconds]')
    ax[1].set_ylabel('Residuals')
    ax[1].set_title('Time domain')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.clf()


# Main function
def main():
    # Parameters
    f0 = 0
    f1 = 300
    T = 10
    A = 3.0
    PSD_AMP = 1e-2
    dt = 0.01 / (2 * f1)
    alpha = 0.2  # Window parameter
    Nf = 2 ** 13  # Number of wavelet frequency bins

    # Generate chirp signal
    y, t = generate_chirp(f0, f1, T, A, dt, alpha)
    ND = len(y)

    # FFT and frequency array
    freq = np.fft.rfftfreq(len(y), dt)
    freq[0] = freq[1]
    df = abs(freq[1] - freq[0])
    y_fft = np.fft.rfft(y)

    # Plot time and frequency domain
    plot_time_freq_domain(t, y, freq, y_fft, f"{OUTDIR}/time_freq_domain.png")

    # Signal in wavelet domain
    print("Executing roundtrip conversions...")
    signal_timeseries = TimeSeries(y, t)
    signal_wavelet_time = from_time_to_wavelet(signal_timeseries, Nf=Nf)
    signal_frequencyseries = FrequencySeries(y_fft, freq=freq)
    signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf=Nf)

    # Plot wavelet domain
    print("Plotting wavelet matrix")
    plot_wavelet(signal_wavelet_freq, f"{OUTDIR}/wavelet_domain.png", freq_range=[f0, f1])

    # Convert back to time and frequency domains
    signal_freq_from_wavelet = from_wavelet_to_freq(signal_wavelet_freq, dt)
    signal_time_from_wavelet = from_wavelet_to_time(signal_wavelet_time, dt)

    # Plot signal reconstructions
    plot_signal_reconstructions(t, y, signal_time_from_wavelet, f"{OUTDIR}/signal_reconstructions.png")

    # Compute residuals
    res_freq = abs(signal_freq_from_wavelet.data - y_fft) ** 2
    res_time = abs(signal_time_from_wavelet.data - y) ** 2

    # Plot residuals
    plot_residuals(freq, res_freq, t, res_time, f"{OUTDIR}/residuals.png")

    # print residual means +/- 1 sigma
    print(f"Frequency domain residual mean: {np.mean(res_freq)} +/- {np.std(res_freq)}")
    print(f"Time domain residual mean: {np.mean(res_time)} +/- {np.std(res_time)}")

    ## SNRs

    # Power spectral density
    PSD = PSD_AMP * np.ones(len(freq))
    psd_wavelet_time = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_time.freq,
        t_grid=signal_wavelet_time.time,
        dt=dt,
    )
    psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_freq.freq,
        t_grid=signal_wavelet_freq.time,
        dt=dt,
    )

    snr2_f = 4 * dt * np.sum(abs(y_fft) ** 2 / (ND * PSD))
    snr2_time_to_wavelet = compute_snr(signal_wavelet_time, psd_wavelet_time) ** 2
    snr2_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2
    print(f"SNR in frequency domain {snr2_f ** 0.5:.2f}", )
    print(f"SNR using time -> wavelet {snr2_time_to_wavelet ** 0.5:.2f}")
    print(f"SNR using freq -> wavelet {snr2_freq_to_wavelet ** 0.5:.2f}")


# Run the main function
if __name__ == "__main__":
    main()
