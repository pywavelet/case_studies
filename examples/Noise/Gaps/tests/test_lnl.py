import matplotlib.pyplot as plt
import numpy as np
from pywavelet.transforms.types import Wavelet

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import TRUES


def test_lnl(plot_dir):
    data = AnalysisData.DEFAULT()
    template = data.htemplate(*TRUES)

    hdiff = data.hwavelet_gapped - template
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    data.hwavelet_gapped.plot(ax=ax[0], show_colorbar=False, label="Data")
    template.plot(ax=ax[1], show_colorbar=False, label="Template")
    hdiff.plot(ax=ax[2], show_colorbar=False, label="Difference")
    plt.subplots_adjust(hspace=0)
    fig.savefig(f"{plot_dir}/lnl.png")

    assert data.hwavelet_gapped == template, "Template and hwavelet not equal!"
    lnl = data.lnl(*TRUES)
    assert lnl == 0, "Lnl not 0 for true params!"


def __plot(hdata, htemplate, lnl, gap, fname):
    # Plot comparison
    # Compute wavelet object of difference between data and template
    diff = Wavelet(hdata.data - htemplate.data, htemplate.time, htemplate.freq)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    hdata.plot(ax=axes[0], label="Data")
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
