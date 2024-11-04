from gap_study_utils.wavelet_data_utils import gap_hwavelet_generator, waveform_generator
from pywavelet.utils import compute_likelihood
from pywavelet.transforms.types import Wavelet
import matplotlib.pyplot as plt
import numpy as np


def test_lnl(plot_dir, test_data):
    ht = test_data.ht
    gap = test_data.gap
    hdata = test_data.wavelet_data
    a_true, ln_f_true, ln_fdot_true = test_data.trues
    htemplate_no_gap = waveform_generator(a_true, np.exp(ln_f_true), np.exp(ln_fdot_true), gap.t, gap.tmax, alpha=test_data.alpha)

    __plot_timeseries([ht, htemplate_no_gap], f"{plot_dir}/wavegenerator_timeseries.png")
    assert htemplate_no_gap.t0 == ht.t0
    assert htemplate_no_gap.dt == ht.dt
    assert htemplate_no_gap.duration == ht.duration
    assert htemplate_no_gap.ND == ht.ND
    assert htemplate_no_gap.data.shape == ht.data.shape
    np.testing.assert_allclose(ht.data, htemplate_no_gap.data, rtol=1e-5)


    htemplate = gap_hwavelet_generator(
        a_true, ln_f_true, ln_fdot_true,
        time=test_data.time, gap=gap, tmax=gap.tmax, Nf=test_data.Nf, alpha=test_data.alpha, filter=test_data.filter
    )
    psd = Wavelet(np.ones_like(htemplate.data), htemplate.time, htemplate.freq)
    lnl = compute_likelihood(hdata, htemplate, psd)

    __plot(hdata, htemplate, lnl, gap, f"{plot_dir}/lnl.png")


    assert lnl == 0, "Lnl not 0 for true params!"

def __plot_timeseries(timeseries, fname):
    n_timeseries = len(timeseries)
    fig, axes = plt.subplots(n_timeseries, 1, figsize=(10, 2*n_timeseries), sharex=True)
    for i, ts in enumerate(timeseries):
        ts.plot(ax=axes[i])
        axes[i].set_title(f"TimeSeries {i}", pad=-10)
    plt.subplots_adjust(hspace=0)
    fig.savefig(fname)


def __plot(hdata, htemplate, lnl,gap, fname):
    # Plot comparison
    diff = hdata - htemplate
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    hdata.plot(ax=axes[0],label="Data")
    htemplate.plot(ax=axes[1], label=f"Template (Lnl = {lnl:.2e})")
    diff.plot(ax=axes[2], label="Data-Template")
    axes[0].set_xlim(0, gap.tmax * 1.1)
    for a in axes:
        a.axvline(gap.gap_start, color="red", linestyle="--", label="Gap")
        a.axvline(gap.gap_end, color="red", linestyle="--")
        a.axvline(gap.tmax, color="green", linestyle="--", label="Tmax")
        a.set_ylim(0.002, 0.007)
    axes[0].legend(loc="lower right")
    plt.subplots_adjust(hspace=0)
    fig.savefig(fname)