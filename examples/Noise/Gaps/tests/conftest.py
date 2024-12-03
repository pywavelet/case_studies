import os

import numpy as np
import pytest

from gap_study_utils.analysis_data import AnalysisData


@pytest.fixture
def plot_dir():
    plt_dir = os.path.join(os.path.dirname(__file__), "out_plots")
    os.makedirs(plt_dir, exist_ok=True)
    return plt_dir


@pytest.fixture
def test_data() -> AnalysisData:
    return AnalysisData.DEFAULT()
