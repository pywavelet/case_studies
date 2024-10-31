from gap_study_utils.mcmc_runner import run_mcmc



def test_mcmc(plot_dir):
    filtering_kwgs = dict(
        noise_realisation=True,
        fmin=0.0001,
        alpha=0.1,
        filter=True,
    )
    filtering_kwgs = dict(
        noise_realisation=False,
        # fmin=0.0001,
        # alpha=0.1,
        filter=False,
    )

    run_mcmc(
        n_iter=250,
        outdir=f"{plot_dir}/gap_mcmc",
        **filtering_kwgs
    )
    run_mcmc(
        n_iter=250,
        gap_range=None,
        outdir=f"{plot_dir}/mcmc",
        **filtering_kwgs
    )
