import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from tqdm import tqdm
from matplotlib import colors


from pywavelet.utils import evolutionary_psd_from_stationary_psd, compute_snr, compute_likelihood
from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.transforms import from_freq_to_wavelet
from gap_study_utils.signal_utils import zero_pad, compute_snr_freq, waveform, generate_padded_signal, waveform_generator
from gap_study_utils.noise_curves import noise_PSD_AE, CornishPowerSpectralDensity
from gap_study_utils.wavelet_data_utils import generate_wavelet_with_gap
from gap_study_utils.gap_funcs import GapWindow
from gap_study_utils.wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap

# Constants
ONE_HOUR = 60 * 60
A_TRUE = 1e-21
F_TRUE = 3e-3
FDOT_TRUE = 1e-8
TMAX = 18.773 * ONE_HOUR
START_GAP = 9 * ONE_HOUR
END_GAP = 11 * ONE_HOUR
NF = 16


OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

def lnl(a, f, fdot, gap, Nf, data, psd, windowing=True, alpha=0.0, filter=True):
    htemplate = gap_hwavelet_generator(a, f, fdot, gap, Nf, windowing=windowing, alpha=alpha, filter=filter)
    return compute_likelihood(data, htemplate, psd)





def gap_hwavelet_generator(a, f, fdot, gap:GapWindow, Nf, windowing=True, alpha=0.0, filter=True)->Wavelet:
    ht = waveform_generator(a, f, fdot, gap.t, alpha)
    return generate_wavelet_with_gap(gap, ht, Nf, windowing=windowing, alpha=alpha, filter=filter)


def generate_data(
    a_true=A_TRUE,
    f_true = F_TRUE,
    fdot_true = FDOT_TRUE,
    start_gap = START_GAP,
    end_gap = END_GAP,
    Nf = NF,
    tmax = TMAX,
    plot=False,
):
    ht, hf = generate_padded_signal(a_true, f_true, fdot_true, tmax)
    h_wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    psd = FrequencySeries(
        data=CornishPowerSpectralDensity(hf.freq),
        freq=hf.freq
    )
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data, psd_f=psd.freq, f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time, dt=hf.dt
    )
    print(f"SNR (hf, no gaps): {compute_snr_freq(hf.data, psd.data, hf.dt, hf.ND)}")
    print(f"SNR (hw, no gaps): {compute_snr(h_wavelet, psd_wavelet)}")

    # Gap data
    gap = GapWindow(ht.time, start_gap, end_gap, )
    chunks = chunk_timeseries(ht, gap)
    hwavelet_with_gap = generate_wavelet_with_gap(
        gap, ht, Nf, windowing=True, alpha=0.0, filter=True
    )
    psd_wavelet_with_gap = gap.apply_nan_gap_to_wavelet(psd_wavelet)
    print(f"SNR (hw, with gaps): {compute_snr(hwavelet_with_gap, psd_wavelet_with_gap)}")

    if plot:
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
        for i in range(2):
            chunks[i].plot(ax=ax[0], color=f"C{i}", label=f"Chunk {i}")
        hwavelet_with_gap.plot(ax=ax[1], show_colorbar=False)
        psd_wavelet_with_gap.plot(ax=ax[2], show_colorbar=False)
        for a in ax:
            a.axvspan(start_gap, end_gap, facecolor='k', alpha=0.2)
            a.axvspan(start_gap, end_gap, edgecolor='k', hatch='/', zorder=10, fill=False)
        ax[0].set_xlim(0, tmax)
        plt.subplots_adjust(hspace=0)
        plt.savefig(os.path.join(OUTDIR, "gaped_data.pdf"), bbox_inches="tight")

    return hwavelet_with_gap, psd_wavelet_with_gap, gap




if __name__ == "__main__":
    generate_data(plot=True)

