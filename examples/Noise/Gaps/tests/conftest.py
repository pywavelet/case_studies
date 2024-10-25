import pytest
import os
import numpy as np
from gap_study_utils.signal_utils import generate_padded_signal
from gap_study_utils.gap_funcs import GapWindow
from gap_study_utils.wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap, gap_hwavelet_generator

from collections import  namedtuple

# Named tuple for the test data
TestData = namedtuple('TestData', ['ht', 'hf', 'gap', 'hwavelet_gap', 'trues'])


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
def test_data()->TestData:
    ht, hf = generate_padded_signal(
        a_true, ln_f_true, ln_fdot_true, tmax
    )
    gap = GapWindow(ht.time, start_gap, end_gap, tmax=tmax)
    h_stiched_wavelet = generate_wavelet_with_gap(gap, ht, Nf)
    return TestData(ht, hf, gap, h_stiched_wavelet, [a_true, ln_f_true, ln_fdot_true])
