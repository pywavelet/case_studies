import matplotlib.pyplot as plt
import numpy as np

from gap_study_utils.gap_window import GapType, GapWindow, TimeSeries


def test_gap_window(plot_dir):
    gap_ranges = [(0.1, 0.2), (0.3, 0.4)]

    t = np.linspace(0, 1, 1000)
    gap_window = GapWindow(t, gap_ranges, 1)

    assert gap_window.num_nans == 200
    assert gap_window.fraction_nans == 0.2
    assert gap_window.gap_len() == 200
    assert len(gap_window) == 1000
    assert gap_window.start_idxs == [100, 300]
    assert gap_window.end_idxs == [199, 399]

    assert gap_window.inside_gap(0.15)
    assert not gap_window.inside_gap(0.25)

    # test gap_windown.non_gap_idxs
    non_gap_idxs = gap_window.non_gap_idxs()
    assert non_gap_idxs[0] == (0, 99)
    assert non_gap_idxs[1] == (200, 299)
    assert non_gap_idxs[2] == (400, 999)

    gap_window.apply_window(np.ones_like(t))
    fig, ax = plt.subplots(1, 1)
    gap_window.plot(ax)
    fig.savefig(f"{plot_dir}/gap_window.png")


def test_gap_window_wavelet(plot_dir):
    Nf = 8
    gap_ranges = [(0.1, 0.2), (0.3, 0.4)]
    t = np.linspace(0, 1, 4096)
    ht = TimeSeries(np.sin(2 * np.pi * 200 * t), t)
    gap_window = GapWindow(t, gap_ranges, 1, type=GapType.STITCH)
    gap_window2 = GapWindow(t, gap_ranges, 1, type=GapType.RECTANGULAR_WINDOW)

    ts = gap_window._chunk_timeseries(ht)
    chunked_wavelets = [
        gap_window.apply_nan_gap_to_wavelet(ts[i].to_wavelet(Nf=Nf))
        for i in range(len(ts))
    ]
    fig, axes = plt.subplots(
        len(chunked_wavelets) + 3, 1, figsize=(15, 10), sharex=True
    )
    for i in range(len(chunked_wavelets)):
        chunked_wavelets[i].plot(
            ax=axes[i], show_colorbar=False, label=f"Chunk {i}\n"
        )
    axes[0].set_xlim(0, gap_window.tmax)
    w = gap_window.gap_n_transform_timeseries(ht, Nf=Nf)
    w.plot(ax=axes[-3], show_colorbar=False, label="Stitch-method\n")
    w2 = gap_window2.gap_n_transform_timeseries(ht, Nf=Nf)
    w2.plot(ax=axes[-2], show_colorbar=False, label="Rectangular-method\n")
    wdiff = w - w2
    wdiff.plot(ax=axes[-1], show_colorbar=False, label="Difference\n")

    for a in axes:
        gap_window.plot(ax=a)
    plt.subplots_adjust(hspace=0)
    fig.savefig(f"{plot_dir}/gap_window_wavelet.png")


def test_repr():
    assert str(GapType.STITCH) == "STITCH"
