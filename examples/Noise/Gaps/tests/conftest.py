import pytest
import os
import numpy as np
from typing import List

from pywavelet.transforms.types import TimeSeries, FrequencySeries, Wavelet
from gap_study_utils.signal_utils import generate_padded_signal
from gap_study_utils.gap_funcs import GapWindow
from gap_study_utils.wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap, gap_hwavelet_generator

from dataclasses import dataclass


@dataclass
class TestData:
    ht: TimeSeries
    hf: FrequencySeries
    gap: GapWindow
    hwavelet_gap: Wavelet
    trues: List[float]
    alpha: float
    windowing: bool
    filter: bool
    Nf: int

    @classmethod
    def from_trues(cls, trues: List[float], windowing: bool, filter: bool, alpha: float, Nf: int, tmax: float,
                   start_gap: float, end_gap: float):
        ht, hf = generate_padded_signal(*trues, tmax, alpha)
        gap = GapWindow(ht.time, start_gap, end_gap, tmax=tmax)
        h_stiched_wavelet = generate_wavelet_with_gap(gap, ht, Nf, windowing=windowing, alpha=alpha, filter=filter)
        return cls(ht, hf, gap, h_stiched_wavelet, trues, alpha, windowing, filter, Nf)


ONE_HOUR = 60 * 60
ONE_DAY = 24 * ONE_HOUR
a_true = 1e-21
ln_f_true = np.log(3e-3)
ln_fdot_true = np.log(1e-8)

# ollie's settings
# tmax = 18.773 * ONE_HOUR
# Nf = 16


Nf = 64
tmax = 4 * ONE_DAY
start_gap = tmax * 0.45
end_gap = start_gap + 2 * ONE_HOUR


@pytest.fixture
def plot_dir():
    plt_dir = os.path.join(os.path.dirname(__file__), "out_plots")
    os.makedirs(plt_dir, exist_ok=True)
    return plt_dir


@pytest.fixture
def test_data() -> TestData:
    return TestData.from_trues(
        [a_true, ln_f_true, ln_fdot_true],
        windowing=True, filter=True, alpha=0.0,
        Nf=Nf, tmax=tmax, start_gap=start_gap, end_gap=end_gap
    )
