from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.random import seed

def test_analysis_data(plot_dir):
    seed(0)
    d = AnalysisData.generate_data(
        noise=False,
        plotfn=f"{plot_dir}/analysis_data.png",
        alpha=0.1,
        highpass_fmin=0.0001,
    )
    d = AnalysisData.generate_data(
        noise=True,
        plotfn=f"{plot_dir}/noisy_analysis_data.png",
        alpha=0.1,
        highpass_fmin=0.0001,
    )