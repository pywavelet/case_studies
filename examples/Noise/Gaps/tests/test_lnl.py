
from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import TRUES
from pywavelet.transforms.types import Wavelet
import matplotlib.pyplot as plt
import numpy as np


def test_lnl(plot_dir):
    data = AnalysisData.generate_data(

    )
    template = data.htemplate(*TRUES)

    hdiff = data.hwavelet - template
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    data.hwavelet.plot(ax=ax[0], show_colorbar=False)
    template.plot(ax=ax[1], show_colorbar=False)
    hdiff.plot(ax=ax[2], show_colorbar=False)
    plt.subplots_adjust(hspace=0)
    fig.savefig(f"{plot_dir}/lnl.png")


    assert data.hwavelet == template, "Template and hwavelet not equal!"
    lnl = data.lnl(*TRUES)
    assert lnl == 0, "Lnl not 0 for true params!"


