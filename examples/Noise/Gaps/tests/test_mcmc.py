from gap_study_utils.constants import DT
from gap_study_utils.mcmc_runner import run_mcmc


def test_mcmc(plot_dir):
    ## Note -- these settings are just to test that the MCMC runs without error
    # the signal is quite shit...
    # for a proper MCMC test (more iterations) look at
    # study/Wavelet_Domain/nan_method/mcmc.py
    kwgs = dict(
        noise_realisation=True,
        alpha=0.1,
        highpass_fmin=0.0001,
        dt=DT,
        tmax=540672,
    )
    run_mcmc(n_iter=200, outdir=f"{plot_dir}/gap_mcmc", **kwgs)

    # No gap
    run_mcmc(
        n_iter=200,
        gap_ranges=None,
        outdir=f"{plot_dir}/basic_mcmc",
        **kwgs,
    )
