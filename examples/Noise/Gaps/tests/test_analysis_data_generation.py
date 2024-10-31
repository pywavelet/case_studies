from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.random import seed

def test_analysis_data(plot_dir):
    seed(0)
    d = AnalysisData.generate_data(
        noise=True,
        plotfn=f"{plot_dir}/noisy_analysis_data.png",
        alpha=0.1,
        fmin=0.01,
    )
