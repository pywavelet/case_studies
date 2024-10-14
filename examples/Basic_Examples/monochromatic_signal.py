import numpy as np
from scipy.signal.windows import tukey
from pywavelet.transforms import from_time_to_wavelet, from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd
from rich.console import Console
from rich.table import Table

# Set up a Rich Console
console = Console()


def generate_signal(f0, A, T, dt):
    """
    Generates a sinusoidal signal with a Tukey window.

    Parameters:
    - f0: float, frequency of the sinusoid [Hz].
    - A: float, amplitude of the sinusoid.
    - T: float, total duration of the signal [seconds].
    - dt: float, sampling interval [seconds].

    Returns:
    - t: numpy array, time vector.
    - y: numpy array, windowed sinusoidal signal.
    """
    t = np.arange(0, T, dt)
    t = t[: 2 ** int(np.log2(len(t)))]  # Round len(t) to the nearest power of 2
    y = A * np.sin(2 * np.pi * f0 * t)  # Generate sinusoidal signal
    window = tukey(len(y), 0.0)  # Tukey window to remove spectral leakage
    y *= window  # Apply the window to the signal
    return t, y


def compute_snr_time_domain(y, PSD_AMP, dt, T, A):
    """
    Computes the SNR in the time domain using Parseval's theorem and analytical formula.

    Parameters:
    - y: numpy array, time-domain signal.
    - PSD_AMP: float, power spectral density amplitude of the noise.
    - dt: float, sampling interval [seconds].
    - T: float, total duration of the signal [seconds].
    - A: float, amplitude of the signal.

    Returns:
    - SNR_t: float, SNR in the time domain.
    - SNR_t_analytical: float, analytically computed SNR in time domain.
    """
    SNR_t = 2 * dt * np.sum(np.abs(y) ** 2 / PSD_AMP)
    SNR_t_analytical = (A ** 2) * T / PSD_AMP
    return np.sqrt(SNR_t), np.sqrt(SNR_t_analytical)


def compute_snr_freq_domain(y, freq, PSD, dt, ND):
    """
    Computes the SNR in the frequency domain using the Fourier-transformed signal.

    Parameters:
    - y: numpy array, time-domain signal.
    - freq: numpy array, frequency vector.
    - PSD: numpy array, power spectral density of the noise.
    - dt: float, sampling interval [seconds].
    - ND: int, number of data points in the signal.

    Returns:
    - SNR_f: float, SNR in the frequency domain.
    """
    y_fft = np.fft.rfft(y)
    SNR_f = 4 * dt * np.sum(np.abs(y_fft) ** 2 / (ND * PSD))
    return np.sqrt(SNR_f), y_fft


def compute_wavelet_domain_snr(y, t, freq, PSD, Nf, dt):
    """
    Computes the SNR in the wavelet domain for both time-to-wavelet and freq-to-wavelet transformations.

    Parameters:
    - y: numpy array, time-domain signal.
    - freq: numpy array, frequency vector.
    - PSD: numpy array, power spectral density of the noise.
    - Nf: int, number of wavelet frequency bins.
    - dt: float, sampling interval [seconds].

    Returns:
    - snr_time_to_wavelet: float, SNR in the wavelet domain (time to wavelet).
    - snr_freq_to_wavelet: float, SNR in the wavelet domain (frequency to wavelet).
    """
    signal_timeseries = TimeSeries(y, t)
    signal_frequencyseries = FrequencySeries(np.fft.rfft(y), freq=freq)

    # Time-to-wavelet transformation
    signal_wavelet_time = from_time_to_wavelet(signal_timeseries, Nf=Nf)
    psd_wavelet_time = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_time.freq,
        t_grid=signal_wavelet_time.time,
        dt=dt,
    )

    # Frequency-to-wavelet transformation
    signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf=Nf)
    psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_freq.freq,
        t_grid=signal_wavelet_freq.time,
        dt=dt,
    )

    # Compute SNRs
    snr_time_to_wavelet = compute_snr(signal_wavelet_time, psd_wavelet_time) ** 2
    snr_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2

    return np.sqrt(snr_time_to_wavelet), np.sqrt(snr_freq_to_wavelet), signal_wavelet_time, signal_wavelet_freq


def display_snr_table(SNR_f, SNR_t, SNR_t_analytical, SNR_time_to_wavelet, SNR_freq_to_wavelet):
    """
    Displays the computed SNRs in a tabular format using rich.

    Parameters:
    - SNR_f: float, SNR in the frequency domain.
    - SNR_t: float, SNR in the time domain.
    - SNR_t_analytical: float, analytical SNR in the time domain.
    - SNR_time_to_wavelet: float, SNR in the wavelet domain (time to wavelet).
    - SNR_freq_to_wavelet: float, SNR in the wavelet domain (frequency to wavelet).
    """
    table = Table(title="SNR Results", title_style="bold cyan")

    table.add_column("Domain", justify="left", style="magenta")
    table.add_column("SNR Value", justify="center", style="green")

    table.add_row("Frequency Domain", f"{SNR_f:.4f}")
    table.add_row("Time Domain (Parseval's Theorem)", f"{SNR_t:.4f}")
    table.add_row("Time Domain (Analytical)", f"{SNR_t_analytical:.4f}")
    table.add_row("Wavelet Domain (Time -> Wavelet)", f"{SNR_time_to_wavelet:.4f}")
    table.add_row("Wavelet Domain (Freq -> Wavelet)", f"{SNR_freq_to_wavelet:.4f}")

    console.print(table)


def main():
    # Parameters
    f0 = 20  # Frequency of the sinusoid
    T = 1000  # Duration of the signal in seconds
    A = 2.0  # Amplitude of the sinusoid
    PSD_AMP = 1e-2  # Amplitude of the noise PSD
    Nf = 512  # Number of wavelet frequency bins

    dt = 0.5 / (2 * f0)  # Sampling interval
    t, y = generate_signal(f0, A, T, dt)

    # Frequency domain properties
    freq = np.fft.rfftfreq(len(y), dt)
    freq[0] = freq[1]  # Prevent division by zero
    PSD = PSD_AMP * np.ones(len(freq))  # Noise PSD

    # Compute SNR in time and frequency domains
    SNR_t, SNR_t_analytical = compute_snr_time_domain(y, PSD_AMP, dt, T, A)
    SNR_f, y_fft = compute_snr_freq_domain(y, freq, PSD, dt, len(y))

    # Compute SNR in wavelet domain
    SNR_time_to_wavelet, SNR_freq_to_wavelet, signal_wavelet_time, signal_wavelet_freq = compute_wavelet_domain_snr(
        y, t, freq, PSD, Nf, dt
    )

    # Display SNR results in a table
    display_snr_table(SNR_f, SNR_t, SNR_t_analytical, SNR_time_to_wavelet, SNR_freq_to_wavelet)

    # Maximum values for wavelet domain
    console.print(
        f"[yellow]Max value at f = {f0} using freq -> wavelet:[/yellow] {np.max(signal_wavelet_freq.data.flatten()):.4f}")
    console.print(
        f"[yellow]Max value at f = {f0} using time -> wavelet:[/yellow] {np.max(signal_wavelet_time.data.flatten()):.4f}")
    console.print(f"[yellow]Hypothesis for true wavelet domain transformation:[/yellow] {A * np.sqrt(2 * Nf):.4f}")


if __name__ == "__main__":
    main()
import numpy as np
from scipy.signal.windows import tukey
from pywavelet.transforms import from_time_to_wavelet, from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.utils import compute_snr, evolutionary_psd_from_stationary_psd
from rich.console import Console
from rich.table import Table

# Set up a Rich Console
console = Console()


def generate_signal(f0, A, T, dt):
    """
    Generates a sinusoidal signal with a Tukey window.

    Parameters:
    - f0: float, frequency of the sinusoid [Hz].
    - A: float, amplitude of the sinusoid.
    - T: float, total duration of the signal [seconds].
    - dt: float, sampling interval [seconds].

    Returns:
    - t: numpy array, time vector.
    - y: numpy array, windowed sinusoidal signal.
    """
    t = np.arange(0, T, dt)
    t = t[: 2 ** int(np.log2(len(t)))]  # Round len(t) to the nearest power of 2
    y = A * np.sin(2 * np.pi * f0 * t)  # Generate sinusoidal signal
    window = tukey(len(y), 0.0)  # Tukey window to remove spectral leakage
    y *= window  # Apply the window to the signal
    return t, y


def compute_snr_time_domain(y, PSD_AMP, dt, T, A):
    """
    Computes the SNR in the time domain using Parseval's theorem and analytical formula.

    Parameters:
    - y: numpy array, time-domain signal.
    - PSD_AMP: float, power spectral density amplitude of the noise.
    - dt: float, sampling interval [seconds].
    - T: float, total duration of the signal [seconds].
    - A: float, amplitude of the signal.

    Returns:
    - SNR_t: float, SNR in the time domain.
    - SNR_t_analytical: float, analytically computed SNR in time domain.
    """
    SNR_t = 2 * dt * np.sum(np.abs(y) ** 2 / PSD_AMP)
    SNR_t_analytical = (A ** 2) * T / PSD_AMP
    return np.sqrt(SNR_t), np.sqrt(SNR_t_analytical)


def compute_snr_freq_domain(y, freq, PSD, dt, ND):
    """
    Computes the SNR in the frequency domain using the Fourier-transformed signal.

    Parameters:
    - y: numpy array, time-domain signal.
    - freq: numpy array, frequency vector.
    - PSD: numpy array, power spectral density of the noise.
    - dt: float, sampling interval [seconds].
    - ND: int, number of data points in the signal.

    Returns:
    - SNR_f: float, SNR in the frequency domain.
    """
    y_fft = np.fft.rfft(y)
    SNR_f = 4 * dt * np.sum(np.abs(y_fft) ** 2 / (ND * PSD))
    return np.sqrt(SNR_f), y_fft


def compute_wavelet_domain_snr(y, t, freq, PSD, Nf, dt):
    """
    Computes the SNR in the wavelet domain for both time-to-wavelet and freq-to-wavelet transformations.

    Parameters:
    - y: numpy array, time-domain signal.
    - freq: numpy array, frequency vector.
    - PSD: numpy array, power spectral density of the noise.
    - Nf: int, number of wavelet frequency bins.
    - dt: float, sampling interval [seconds].

    Returns:
    - snr_time_to_wavelet: float, SNR in the wavelet domain (time to wavelet).
    - snr_freq_to_wavelet: float, SNR in the wavelet domain (frequency to wavelet).
    """
    signal_timeseries = TimeSeries(y, t)
    signal_frequencyseries = FrequencySeries(np.fft.rfft(y), freq=freq)

    # Time-to-wavelet transformation
    signal_wavelet_time = from_time_to_wavelet(signal_timeseries, Nf=Nf)
    psd_wavelet_time = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_time.freq,
        t_grid=signal_wavelet_time.time,
        dt=dt,
    )

    # Frequency-to-wavelet transformation
    signal_wavelet_freq = from_freq_to_wavelet(signal_frequencyseries, Nf=Nf)
    psd_wavelet_freq = evolutionary_psd_from_stationary_psd(
        psd=PSD,
        psd_f=freq,
        f_grid=signal_wavelet_freq.freq,
        t_grid=signal_wavelet_freq.time,
        dt=dt,
    )

    # Compute SNRs
    snr_time_to_wavelet = compute_snr(signal_wavelet_time, psd_wavelet_time) ** 2
    snr_freq_to_wavelet = compute_snr(signal_wavelet_freq, psd_wavelet_freq) ** 2

    return np.sqrt(snr_time_to_wavelet), np.sqrt(snr_freq_to_wavelet), signal_wavelet_time, signal_wavelet_freq


def display_snr_table(SNR_f, SNR_t, SNR_t_analytical, SNR_time_to_wavelet, SNR_freq_to_wavelet):
    """
    Displays the computed SNRs in a tabular format using rich.

    Parameters:
    - SNR_f: float, SNR in the frequency domain.
    - SNR_t: float, SNR in the time domain.
    - SNR_t_analytical: float, analytical SNR in the time domain.
    - SNR_time_to_wavelet: float, SNR in the wavelet domain (time to wavelet).
    - SNR_freq_to_wavelet: float, SNR in the wavelet domain (frequency to wavelet).
    """
    table = Table(title="SNR Results", title_style="bold cyan")

    table.add_column("Domain", justify="left", style="magenta")
    table.add_column("SNR Value", justify="center", style="green")

    table.add_row("Frequency Domain", f"{SNR_f:.4f}")
    table.add_row("Time Domain (Parseval's Theorem)", f"{SNR_t:.4f}")
    table.add_row("Time Domain (Analytical)", f"{SNR_t_analytical:.4f}")
    table.add_row("Wavelet Domain (Time -> Wavelet)", f"{SNR_time_to_wavelet:.4f}")
    table.add_row("Wavelet Domain (Freq -> Wavelet)", f"{SNR_freq_to_wavelet:.4f}")

    console.print(table)


def main():
    # Parameters
    f0 = 20  # Frequency of the sinusoid
    T = 1000  # Duration of the signal in seconds
    A = 2.0  # Amplitude of the sinusoid
    PSD_AMP = 1e-2  # Amplitude of the noise PSD
    Nf = 512  # Number of wavelet frequency bins

    dt = 0.5 / (2 * f0)  # Sampling interval
    t, y = generate_signal(f0, A, T, dt)

    # Frequency domain properties
    freq = np.fft.rfftfreq(len(y), dt)
    freq[0] = freq[1]  # Prevent division by zero
    PSD = PSD_AMP * np.ones(len(freq))  # Noise PSD

    # Compute SNR in time and frequency domains
    SNR_t, SNR_t_analytical = compute_snr_time_domain(y, PSD_AMP, dt, T, A)
    SNR_f, y_fft = compute_snr_freq_domain(y, freq, PSD, dt, len(y))

    # Compute SNR in wavelet domain
    SNR_time_to_wavelet, SNR_freq_to_wavelet, signal_wavelet_time, signal_wavelet_freq = compute_wavelet_domain_snr(
        y, t, freq, PSD, Nf, dt
    )

    # Display SNR results in a table
    display_snr_table(SNR_f, SNR_t, SNR_t_analytical, SNR_time_to_wavelet, SNR_freq_to_wavelet)

    # Maximum values for wavelet domain
    console.print(
        f"[yellow]Max value at f = {f0} using freq -> wavelet:[/yellow] {np.max(signal_wavelet_freq.data.flatten()):.4f}")
    console.print(
        f"[yellow]Max value at f = {f0} using time -> wavelet:[/yellow] {np.max(signal_wavelet_time.data.flatten()):.4f}")
    console.print(f"[yellow]Hypothesis for true wavelet domain transformation:[/yellow] {A * np.sqrt(2 * Nf):.4f}")


if __name__ == "__main__":
    main()
