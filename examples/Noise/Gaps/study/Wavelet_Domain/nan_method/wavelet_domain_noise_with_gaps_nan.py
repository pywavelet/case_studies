import os
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from constants import *
from matplotlib import colors
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.utils import (
    compute_likelihood,
    compute_snr,
    evolutionary_psd_from_stationary_psd,
)
from scipy.signal.windows import tukey
from tqdm import tqdm

from gap_study_utils.gap_funcs import GapWindow
from gap_study_utils.noise_curves import (
    CornishPowerSpectralDensity,
    noise_PSD_AE,
)
from gap_study_utils.signal_utils import (
    compute_snr_freq,
    generate_padded_signal,
)
from gap_study_utils.wavelet_data_utils import (
    chunk_timeseries,
    gap_hwavelet_generator,
    generate_wavelet_with_gap,
)


def lnl(
    a: float,
    ln_f: float,
    ln_fdot: float,
    gap: GapWindow,
    Nf: int,
    data: Wavelet,
    psd: Wavelet,
    windowing: bool = False,
    alpha: float = 0.0,
    filter: bool = False,
) -> float:
    htemplate = gap_hwavelet_generator(
        a,
        ln_f,
        ln_fdot,
        gap,
        Nf,
        windowing=windowing,
        alpha=alpha,
        filter=filter,
    )
    return compute_likelihood(data, htemplate, psd)


def generate_stat_noise(
    ht: TimeSeries, psd: FrequencySeries, seed_no: int = 0, TD: bool = True
) -> Union[TimeSeries, FrequencySeries]:
    """
    Inputs  ht: TimeSeries
            psd: FrequencySeries

    Outputs stationary noise in time domain as TimeSeries object.
    """

    np.random.seed(seed_no)
    dt = ht.dt  # Extract sampling interval
    ND = len(ht.time)  # Extract length of time-series
    variance_f = (
        ND * psd.data / (4 * dt)
    )  # Variance of stationary noise process

    # Generate noise in frequency domain
    real_noise_f = np.random.normal(0, np.sqrt(variance_f))
    imag_noise_f = np.random.normal(0, np.sqrt(variance_f))
    noise_f = real_noise_f + 1j * imag_noise_f

    # Real process (positive frequencies).
    noise_f[0] = np.sqrt(2) * noise_f[0].real
    noise_f[-1] = np.sqrt(2) * noise_f[-1].real

    # Convert to time-series
    if TD == True:
        noise_t = np.fft.irfft(noise_f)
        # Cast as TimeSeries object
        return TimeSeries(noise_t, ht.time)
    else:
        return FrequencySeries(noise_f, psd.freq)


def generate_data(
    a_true: float = A_TRUE,
    ln_f_true: float = LN_F_TRUE,
    ln_fdot_true: float = LN_FDOT_TRUE,
    start_gap: float = START_GAP,
    end_gap: float = END_GAP,
    Nf: int = NF,
    tmax: float = TMAX,
    noise_realisation: bool = False,
    seed_no: int = 11_07_1993,
    windowing: bool = False,
    alpha: float = 0.0,
    filter: bool = False,
    plotfn: str = "",
) -> Tuple[Wavelet, Wavelet, GapWindow]:
    """
    Generate data with gaps and corresponding PSD in the wavelet domain.

    :param a_true: Amplitude of the signal.
    :param ln_f_true: Natural logarithm of the frequency.
    :param ln_fdot_true: Natural logarithm of the frequency derivative.
    :param start_gap: Start time of the gap (in seconds).
    :param end_gap: End time of the gap (in seconds).
    :param Nf: Number of frequency bins.
    :param tmax: Maximum time for the signal (in seconds).
    :param noise_realisation: Flag to include noise realisation.
    :param seed_no: Seed number for random noise generation.
    :param windowing: Flag to apply windowing to the signal.
    :param alpha: Alpha parameter for the windowing function.
    :param filter: Flag to apply a high-pass filter.
    :param plotfn: Filename to save the plot.

    :return: Wavelet data, Wavelet PSD, and GapWindow
    :type: Tuple[Wavelet, Wavelet, GapWindow]
     Tuple containing the wavelet-transformed data with gaps, the PSD with gaps, and the gap window.
    """

    if noise_realisation is True:
        noise_flag = 1.0
    else:
        noise_flag = 0.0

    ht, hf = generate_padded_signal(a_true, ln_f_true, ln_fdot_true, tmax)
    h_wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    psd = FrequencySeries(
        data=CornishPowerSpectralDensity(hf.freq), freq=hf.freq
    )
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time,
        dt=hf.dt,
    )
    print(
        f"SNR (hf, no gaps): {compute_snr_freq(hf.data, psd.data, hf.dt, hf.ND)}"
    )
    print(f"SNR (hw, no gaps): {compute_snr(h_wavelet, psd_wavelet)}")

    # Generate data
    noise_t = generate_stat_noise(ht, psd, seed_no=seed_no)

    data_stream = TimeSeries(ht.data + noise_flag * noise_t.data, ht.time)
    # Gap data
    gap = GapWindow(data_stream.time, start_gap, end_gap, tmax=tmax)
    chunks = chunk_timeseries(data_stream, gap)
    h_wavelet_with_gap = generate_wavelet_with_gap(
        gap, ht, Nf, windowing=windowing, alpha=alpha, filter=filter
    )

    data_wavelet_with_gap = generate_wavelet_with_gap(
        gap, data_stream, Nf, windowing=windowing, alpha=alpha, filter=filter
    )
    psd_wavelet_with_gap = gap.apply_nan_gap_to_wavelet(psd_wavelet)
    print(
        f"SNR (hw, with gaps): {compute_snr(h_wavelet_with_gap, psd_wavelet_with_gap)}"
    )

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    h_wavelet.plot(ax=axes[0], show_colorbar=False, detailed_axes=True)
    psd_wavelet.plot(ax=axes[1], show_colorbar=False, detailed_axes=True)
    data_wavelet_with_gap.plot(
        ax=axes[2], show_colorbar=False, detailed_axes=True
    )
    psd_wavelet.plot(ax=axes[3], show_colorbar=False, detailed_axes=True)
    plt.subplots_adjust(hspace=0)
    for a in axes:
        a.axvline(tmax, color="red", linestyle="--", label="Gap")
        a.set_xlabel("")
        a.set_ylabel("")
    axes[0].set_xlim(0, tmax * 1.1)
    plt.savefig(os.path.join(OUTDIR, "wavelet_debug.pdf"), bbox_inches="tight")

    if plotfn:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        for i in range(2):
            chunks[i].plot(ax=ax[0], color=f"C{i}", label=f"Chunk {i}")
        data_wavelet_with_gap.plot(ax=ax[1], show_colorbar=False)
        psd_wavelet_with_gap.plot(ax=ax[2], show_colorbar=False)
        h_wavelet.plot(ax=ax[3], show_colorbar=False)
        for a in ax:
            a.axvspan(start_gap, end_gap, facecolor="k", alpha=0.2)
            a.axvspan(
                start_gap,
                end_gap,
                edgecolor="k",
                hatch="/",
                zorder=10,
                fill=False,
            )
        ax[0].set_xlim(0, tmax * 1.1)
        plt.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(OUTDIR, plotfn), bbox_inches="tight")

    return data_wavelet_with_gap, psd_wavelet_with_gap, gap


# def generate_data_ollie(
#     a_true=A_TRUE,
#     ln_f_true = LN_F_TRUE,
#     ln_fdot_true = LN_FDOT_TRUE,
#     start_gap = START_GAP,
#     end_gap = END_GAP,
#     Nf = NF,
#     tmax = TMAX,
#     plotfn="",
# ):
#     ht, hf = generate_padded_signal(a_true, ln_f_true, ln_fdot_true, tmax)
#     h_wavelet = from_freq_to_wavelet(hf, Nf=Nf)

#     noise_t = 0 * np.zeros(len(ht))

#     data_t = TimeSeries(ht.data + noise_t, ht.time)

#     gap = GapWindow(data_t.time, start_gap, end_gap, tmax=tmax)
#     data_wavelet_with_gap = generate_wavelet_with_gap(
#         gap, data_t, Nf, windowing=False, alpha=0.0, filter=False
#     )

#     psd = FrequencySeries(
#         data=CornishPowerSpectralDensity(hf.freq),
#         freq=hf.freq
#     )

#     psd_wavelet = evolutionary_psd_from_stationary_psd(
#         psd=psd.data, psd_f=psd.freq, f_grid=h_wavelet.freq,
#         t_grid=h_wavelet.time, dt=hf.dt
#     )

#     psd_wavelet_with_gap = gap.apply_nan_gap_to_wavelet(psd_wavelet)
#     print(f"SNR (hf, no gaps): {compute_snr_freq(hf.data, psd.data, hf.dt, hf.ND)}")
#     print(f"SNR (hw, no gaps): {compute_snr(h_wavelet, psd_wavelet)}")

#     # Gap data
#     print(f"SNR (hw, with gaps): {compute_snr(data_wavelet_with_gap, psd_wavelet_with_gap)}")

#     return data_wavelet_with_gap, psd_wavelet_with_gap, gap


if __name__ == "__main__":
    generate_data(plotfn="gaped_data.png")
