import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from tqdm import tqdm


from pywavelet.utils import evolutionary_psd_from_stationary_psd, compute_snr, compute_likelihood
from pywavelet.transforms.types import FrequencySeries
from pywavelet.transforms import from_freq_to_wavelet
from gap_study_utils.signal_utils import zero_pad, inner_prod, waveform
from gap_study_utils.noise_curves import noise_PSD_AE, CornishPowerSpectralDensity
from gap_study_utils.wavelet_data_utils import stitch_together_data_wavelet

# Constants
ONE_HOUR = 60 * 60

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)



def gap_routine_nan(t, start_window, end_window, delta_t=10):
    """
    Insert NaN values in a given time window to simulate gaps in the data.

    Parameters:
    - t: Time array.
    - start_window: Start of the gap window in hours.
    - end_window: End of the gap window in hours.
    - delta_t: Time step.

    Returns:
    - nan_window: List with 1s for valid data and NaNs for missing data in the gap.
    """
    start_window *= ONE_HOUR
    end_window *= ONE_HOUR
    nan_window = [np.nan if start_window < time < end_window else 1 for time in t]
    return nan_window


def compute_snr_freq(h_f, PSD, delta_t, N):
    """
    Compute the optimal matched filtering SNR in the frequency domain.

    Parameters:
    - h_f: Fourier transform of the signal.
    - PSD: Power spectral density.
    - delta_t: Sampling interval.
    - N: Length of time series.

    Returns:
    - snr: SNR value.
    """
    SNR2 = inner_prod(h_f, h_f, PSD, delta_t, N)
    return np.sqrt(SNR2)



def plot_waveform_with_gaps(t_pad, h_pad_w, start_gap, end_gap, plot_dir="../plots"):
    """
    Plot the waveform with gaps and save the plot.

    Parameters:
    - t_pad: Padded time array.
    - h_pad_w: Waveform with NaN gaps.
    - plot_dir: Directory to save the plot.
    """
    plt.plot(t_pad / ONE_HOUR, h_pad_w)
    plt.xlabel(r'Time t [Hrs]')
    plt.ylabel(r'Signal')
    plt.title(f'Waveform with Gaps ({start_gap}-{end_gap} hrs)')
    plt.grid()
    os.makedirs(plot_dir, exist_ok=True)
    plt.savefig(os.path.join(plot_dir, "waveform_nan.pdf"), bbox_inches="tight")
    plt.clf()

def generate_likelihood(a_range, t, w_t, t_pad, delta_t, start_gap, end_gap, Nf, data_set_wavelet_stitched,
                        Wavelet_Matrix_with_nans, f_true, fdot_true, a_true, TDI):
    """
    Generate and plot the likelihood for different amplitude values in the wavelet domain.

    Parameters:
    - a_range: Range of amplitude values to test.
    - t: Time array.
    - w_t: Weighting window to apply NaNs.
    - t_pad: Padded time array.
    - delta_t: Sampling interval.
    - start_gap: Start of gap in hours.
    - end_gap: End of gap in hours.
    - Nf: Number of frequency bins in wavelet transform.
    - data_set_wavelet_stitched: Stitched data in the wavelet domain.
    - Wavelet_Matrix_with_nans: PSD matrix in wavelet domain with NaNs.
    - f_true: True frequency.
    - fdot_true: True frequency derivative.
    """
    llike_vec = []
    for a_val in tqdm(a_range):
        h_prop = waveform(a_val, f_true, fdot_true, t)
        h_prop_pad = zero_pad(h_prop * tukey(len(h_prop), alpha=0.0))
        h_prop_wavelet, _ = stitch_together_data_wavelet(w_t, t_pad, h_prop_pad, Nf, delta_t, start_gap, end_gap,
                                                         windowing=True, alpha=0.0, filter=True)
        llike_val = compute_likelihood(data_set_wavelet_stitched, h_prop_wavelet, Wavelet_Matrix_with_nans)
        llike_vec.append(llike_val)

    # Convert to array and plot
    llike_vec_array = np.array(llike_vec)
    plt.plot(a_range, np.exp(llike_vec_array), '*')
    plt.plot(a_range, np.exp(llike_vec_array))
    plt.axvline(x=a_true, c='red', linestyle='--', label="truth")
    plt.legend()
    plt.xlabel(r'Amplitude values')
    plt.ylabel(r'Likelihood')
    plt.title(f'Likelihood with {TDI} noise and Nf = {Nf}')
    plt.savefig(os.path.join(OUTDIR, f"likelihood_{TDI}_Nf_{Nf}.pdf"), bbox_inches="tight")


def main(
    a_true=1e-21,
    f_true = 3e-3,
    fdot_true = 1e-8,
    start_gap = 4,
    end_gap = 6,
    Nf = 16,
    tmax = 10 * ONE_HOUR,
):
    fs = 2 * f_true
    delta_t = np.floor(0.4 / fs)
    t = np.arange(0, tmax, delta_t)
    N = int(2 ** np.ceil(np.log2(len(t))))

    # Generate signal and apply tapering
    h_t = waveform(a_true, f_true, fdot_true, t)
    h_t_pad = zero_pad(h_t * tukey(len(h_t), alpha=0.0))
    t_pad = np.arange(0, len(h_t_pad) * delta_t, delta_t)

    # Frequency domain
    h_true_f = np.fft.rfft(h_t_pad)
    freq = np.fft.rfftfreq(N, delta_t)
    freq[0] = freq[1]

    PSD = CornishPowerSpectralDensity(freq)
    SNR = compute_snr_freq(h_true_f, PSD, delta_t, N)
    print(f"SNR of source: {SNR}")

    # Wavelet domain
    signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
    psd = FrequencySeries(data=PSD, freq=freq)
    h_wavelet = from_freq_to_wavelet(signal_f_series, Nf=Nf)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data, psd_f=psd.freq, f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time, dt=delta_t
    )
    SNR_wavelet = compute_snr(h_wavelet, psd_wavelet.data)
    print(f"SNR in wavelet domain: {SNR_wavelet}")

    # Gap handling and plotting
    w_t = gap_routine_nan(t_pad, start_gap, end_gap, delta_t)
    h_pad_w = w_t * h_t_pad
    plot_waveform_with_gaps(t_pad, h_pad_w, start_gap, end_gap, plot_dir=OUTDIR)

    # Stitched data and likelihood
    h_approx_stitched_data, mask = stitch_together_data_wavelet(w_t, t_pad, h_t_pad, Nf, delta_t, start_gap, end_gap,
                                                                windowing=True, alpha=0.0, filter=True)
    Wavelet_Matrix_with_nans = psd_wavelet.data.copy()
    Wavelet_Matrix_with_nans[:, ~mask] = np.nan

    # Likelihood calculation and plotting
    precision = a_true / np.sqrt(np.nansum(h_wavelet.data ** 2 / Wavelet_Matrix_with_nans))
    a_range = np.linspace(a_true - 5 * precision, a_true + 5 * precision, 200)
    generate_likelihood(a_range, t, w_t, t_pad, delta_t, start_gap, end_gap, Nf, h_approx_stitched_data,
                        Wavelet_Matrix_with_nans, f_true, fdot_true, a_true, TDI="Cornish")


if __name__ == "__main__":
    main()
