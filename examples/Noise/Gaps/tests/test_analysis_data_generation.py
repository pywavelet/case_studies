from gap_study_utils.analysis_data import AnalysisData, F_TRUE
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
    outdir = f"{plot_dir}/analysis_data"
    os.makedirs(outdir, exist_ok=True)
    AnalysisData.generate_data(
        noise=noise,
        plotfn=f"{outdir}/{gap_type}_noise[{noise}].png",
        dt=dt,
        gap_type=gap_type,
    )