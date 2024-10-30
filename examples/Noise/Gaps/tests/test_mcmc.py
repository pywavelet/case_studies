from gap_study_utils.mcmc_runner import run_mcmc

def test_mcmc(plot_dir):
    run_mcmc(
        n_iter=50,
        outdir=f"{plot_dir}/mcmc",
    )
    run_mcmc(
        n_iter=50,
        gap_range=None,
        outdir=f"{plot_dir}/gapmcmc"
    )
