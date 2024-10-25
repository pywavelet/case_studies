from case_studies.examples.LISA.lisa_wavelet_mcmc import h_wavelet
from case_studies.examples.Noise.Gaps.tests.conftest import ln_f_true, ln_fdot_true
from gap_study_utils.wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap, gap_hwavelet_generator
from gap_study_utils.wavelet_data_utils import from_freq_to_wavelet
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.utils import compute_likelihood
from pywavelet.transforms.types import TimeSeries, Wavelet
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np


def test_lnl(plot_dir, test_data):
    ht = test_data.ht
    gap = test_data.gap
    hdata = test_data.hwavelet_gap
    a_true, ln_f_true, ln_fdot_true = test_data.trues

    htemplate = gap_hwavelet_generator(a_true, ln_f_true, ln_fdot_true, gap, Nf)
    psd = Wavelet(np.ones_like(htemplate.data), htemplate.time, htemplate.freq)
    lnl = compute_likelihood(hdata, htemplate, psd)

    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    hdata.plot(ax=axes[0], label="Data")
    htemplate.plot(ax=axes[1], label="Template")
    # suptitle LnL
    fig.suptitle(f"LnL: {lnl}")
    plt.tight_layout()
    fig.savefig(f"{plot_dir}/lnl.png")

    assert lnl == 0, "Lnl not 0 for true params!"