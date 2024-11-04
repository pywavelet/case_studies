
from gap_study_utils.wavelet_data_utils import chunk_timeseries
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.utils import compute_likelihood
from pywavelet.transforms.types import TimeSeries, Wavelet, FrequencySeries
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List


def test_chunked_timeseries(plot_dir, test_data):
    ht = test_data.ht
    gaps = test_data.gaps
    h_stiched_wavelet = test_data.wavelet_data

    chunks = chunk_timeseries(ht, gaps, windowing_alpha=0.1, filter=True)
    chunksf = [c.to_frequencyseries() for c in chunks]
    chunkd_wavelets = [from_freq_to_wavelet(chunk.to_frequencyseries(), 16) for chunk in chunks]


    assert len(chunks) == len(gaps) + 1
    # assert chunks have no nans
    for c in chunks:
        assert np.all(~np.isnan(c.data))
    # assert gap.gap_len() + len(chunks[0].time) + len(chunks[1].time) == len(ht.time)
    for i in range(len(chunks)):
        assert chunkd_wavelets[i].time[0] == chunks[i].time[0]

    _plot(chunks, chunksf, h_stiched_wavelet, plot_dir)


def _plot(
        chunks:List[TimeSeries],
        chunksf:List[FrequencySeries],
        h_stiched_wavelet:Wavelet,
        plot_dir:str
):

    fig, axes = plt.subplots(3,1, figsize=(10, 10))
    ax_time = axes[0]
    ax_freq = axes[1]
    ax_stiched_wavelet = axes[2]
    for i in range(len(chunks)):
        chunks[i].plot(ax=ax_time, color=f"C{i}", label=f"Chunk {i}")
        chunksf[i].plot_periodogram(ax=ax_freq, color=f"C{i}", label=f"Chunk {i}")
    h_stiched_wavelet.plot(ax=ax_stiched_wavelet, show_colorbar=False)
    ax_time.set_xlim(0, chunks[-1].tend)
    ax_stiched_wavelet.set_xlim(0, chunks[-1].tend)
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/chunked_timeseries.png")
    plt.close(fig)


