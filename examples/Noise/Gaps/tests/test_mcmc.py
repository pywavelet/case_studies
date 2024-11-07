from gap_study_utils.mcmc_runner import run_mcmc



def test_mcmc(plot_dir):
    filtering_kwgs = dict(
        noise_realisation=True,
        alpha=0.1,
        highpass_fmin=0.0001,
        dt=10,
        tmax=655360,
    )

    run_mcmc(
        n_iter=250,
        outdir=f"{plot_dir}/gap_mcmc",
        **filtering_kwgs
    )
    run_mcmc(
        n_iter=250,
        gap_ranges=None,
        outdir=f"{plot_dir}/mcmc",
        **filtering_kwgs
    )
