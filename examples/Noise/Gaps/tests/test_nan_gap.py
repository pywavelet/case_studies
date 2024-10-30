
from gap_study_utils.wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap, gap_hwavelet_generator, GapWindow
from gap_study_utils.wavelet_data_utils import from_freq_to_wavelet
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.utils import compute_likelihood
from pywavelet.transforms.types import TimeSeries, Wavelet, FrequencySeries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List


def test_chunked_timeseries(plot_dir, test_data):
    ht = test_data.ht
    gap = test_data.gap
    h_stiched_wavelet = test_data.hwavelet
    a_true, ln_f_true, ln_fdot_true = test_data.trues
    chunks = chunk_timeseries(ht, gap)
    chunksf = [c.to_frequencyseries() for c in chunks]
    chunkd_wavelets = [from_freq_to_wavelet(chunk.to_frequencyseries(), 16) for chunk in chunks]

    assert gap.start_idx < gap.end_idx
    assert gap.start_idx != 0
    assert gap.end_idx != len(ht.time) - 1
    assert len(chunks) == 2
    # assert chunks have no nans
    for c in chunks:
        assert np.all(~np.isnan(c.data))
    # assert gap.gap_len() + len(chunks[0].time) + len(chunks[1].time) == len(ht.time)
    for i in range(2):
        assert chunkd_wavelets[i].time[0] == chunks[i].time[0]

    _plot(chunks, chunksf, chunkd_wavelets, h_stiched_wavelet, gap, plot_dir)


def _plot(
        chunks:List[TimeSeries], chunksf:List[FrequencySeries], chunkd_wavelets:List[Wavelet],
        h_stiched_wavelet:Wavelet, gap:GapWindow, plot_dir:str):
    # make gridspec  3 rows, 2 columns (first two rows are 1 plot each, last row is 2 plots)
    g = GridSpec(4, 2)
    fig = plt.figure(figsize=(10, 10))
    ax_time = fig.add_subplot(g[0, :])
    ax_freq = fig.add_subplot(g[1, :])
    ax_wavelet1 = fig.add_subplot(g[2, 0])
    ax_wavelet2 = fig.add_subplot(g[2, 1])
    ax_wavelet = [ax_wavelet1, ax_wavelet2]
    ax_stiched_wavelet = fig.add_subplot(g[3, :])
    for i in range(2):
        chunks[i].plot(ax=ax_time, color=f"C{i}", label=f"Chunk {i}")
        chunksf[i].plot_periodogram(ax=ax_freq, color=f"C{i}", label=f"Chunk {i}")
        chunkd_wavelets[i].plot(ax=ax_wavelet[i], show_colorbar=False)
    h_stiched_wavelet.plot(ax=ax_stiched_wavelet, show_colorbar=False)
    ax_time.set_xlim(0, gap.tmax)
    ax_time.axvline(gap.gap_start, color="red", linestyle="--", label="Gap")
    ax_time.axvline(gap.gap_end, color="red", linestyle="--")
    ax_wavelet1.set_xlim(0, gap.tmax)
    ax_wavelet2.set_xlim(0, gap.tmax)
    ax_stiched_wavelet.set_xlim(0, gap.tmax)
    ax_stiched_wavelet.axvline(gap.gap_start, color="red", linestyle="--", label="Gap")
    ax_stiched_wavelet.axvline(gap.gap_end, color="red", linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/chunked_timeseries.png")
    plt.close(fig)


