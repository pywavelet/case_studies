from gap_study_utils.signal_utils import generate_padded_signal
from gap_study_utils.gap_funcs import GapWindow
from gap_study_utils.wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap
from pywavelet.transforms import from_freq_to_wavelet
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

ONE_HOUR = 60 * 60
a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

Nf = 16
tmax = 18.773 * ONE_HOUR
start_gap = tmax  * 0.45
end_gap = start_gap + 2 * ONE_HOUR


def test_chunked_timeseries(plot_dir):
    ht, hf = generate_padded_signal(
        a_true, f_true, fdot_true, tmax
    )
    gap = GapWindow(ht.time, start_gap, end_gap, )

    assert gap.start_idx < gap.end_idx
    assert gap.start_idx != 0
    assert gap.end_idx != len(ht.time) - 1


    chunks = chunk_timeseries(ht, gap)
    chunksf = [c.to_frequencyseries() for c in chunks]
    assert len(chunks) == 2
    # assert chunks have no nans
    for c in chunks:
        assert np.all(~np.isnan(c.data))
    # assert gap.gap_len() + len(chunks[0].time) + len(chunks[1].time) == len(ht.time)


    assert len(chunks) == 2
    chunkd_wavelets = [from_freq_to_wavelet(chunk.to_frequencyseries(), 16) for chunk in chunks]

    for i in range(2):
        assert chunkd_wavelets[i].time[0] == chunks[i].time[0]

    h_stiched_wavelet  =  generate_wavelet_with_gap(
        gap, ht, Nf
    )


    # make gridspec  3 rows, 2 columns (first two rows are 1 plot each, last row is 2 plots)
    g = GridSpec(4, 2)
    fig = plt.figure(figsize=(10, 10))
    ax_time = fig.add_subplot(g[0, :])
    ax_freq = fig.add_subplot(g[1, :])
    ax_wavelet1= fig.add_subplot(g[2, 0])
    ax_wavelet2 = fig.add_subplot(g[2, 1])
    ax_wavelet = [ax_wavelet1, ax_wavelet2]
    ax_stiched_wavelet = fig.add_subplot(g[3, :])
    for i in range(2):
        chunks[i].plot(ax=ax_time, color=f"C{i}", label=f"Chunk {i}")
        chunksf[i].plot_periodogram(ax=ax_freq, color=f"C{i}", label=f"Chunk {i}")
        chunkd_wavelets[i].plot(ax=ax_wavelet[i], show_colorbar=False)
    h_stiched_wavelet.plot(ax=ax_stiched_wavelet, show_colorbar=False)
    ax_time.set_xlim(0, tmax)
    ax_time.axvline(start_gap, color="red", linestyle="--", label="Gap")
    ax_time.axvline(end_gap, color="red", linestyle="--")
    ax_wavelet1.set_xlim(0, tmax)
    ax_wavelet2.set_xlim(0, tmax)
    ax_stiched_wavelet.set_xlim(0, tmax)
    ax_stiched_wavelet.axvline(start_gap, color="red", linestyle="--", label="Gap")
    ax_stiched_wavelet.axvline(end_gap, color="red", linestyle="--")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/chunked_timeseries.png")



