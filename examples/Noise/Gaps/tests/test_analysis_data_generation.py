import numpy as np

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import F_TRUE, FDOT_TRUE, A_TRUE, TMAX, GAP_RANGES
from gap_study_utils.signal_utils import waveform
from gap_study_utils.random import seed
from gap_study_utils.gap_window import GapType
import pytest
import os


# parameterize gap_type and noise
@pytest.mark.parametrize("gap_type", GapType.all_types())
@pytest.mark.parametrize("noise", [True, False])
def test_analysis_data(plot_dir, gap_type, noise):
    seed(0)
    dt = 10
    tmax=655360
    gap_ranges = GAP_RANGES
    outdir = f"{plot_dir}/analysis_data"
    os.makedirs(outdir, exist_ok=True)

    AnalysisData(
        data_kwargs=dict(dt=dt, noise=noise, tmax=TMAX),
        gap_kwargs=dict(
            type=gap_type,
            gap_ranges=gap_ranges
        ),
        waveform_generator=waveform,
        waveform_parameters=[A_TRUE, F_TRUE, FDOT_TRUE],
        plotfn=f"{outdir}/analysis_data_{gap_type}_{noise}.png"
    )