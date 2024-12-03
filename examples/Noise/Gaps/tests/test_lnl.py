
from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import TRUES
from pywavelet.transforms.types import Wavelet
import matplotlib.pyplot as plt
import numpy as np


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


