import emcee
from .analysis_data import AnalysisData
from .bayesian_functions import sample_prior, log_posterior, PRIOR, CENTERED_PRIOR
from .constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, ONE_HOUR, TRUES
from .plotting import plot_corner, make_mcmc_trace_gif, plot_mcmc_summary
from .random import seed
from multiprocessing import (get_context, cpu_count)
import numpy as np
import os
import arviz as az


def run_mcmc(
        true_params=[A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
        gap_range=[START_GAP, END_GAP],
        Nf=NF,
        tmax=TMAX,
        alpha=0.0,
        filter=False,
        noise_realisation=False,
        n_iter=2500,
        nwalkers=32,
        random_seed=None,
        outdir="out_mcmc",
):
    """
    Run MCMC on the data generated with the given parameters.

    :param true_params: True parameters of the signal.
    :param gap_range: [Start, end] time of the gap (in seconds).
    :param
    :param Nf: Number of frequency bins.
    :param tmax: Maximum time for the signal (in seconds).
    :param alpha: Alpha parameter for the windowing function.
    :param filter: Flag to apply a high-pass filter.
    :param noise_realisation: Flag to include noise realisation.
    :param n_iter: Number of iterations for the MCMC.
    :param nwalkers: Number of walkers for the MCMC.
    :param random_seedrandom_seed: Seed number for data_generation + MCMC.
    :param outdir: Output directory to save the chain + plots.
    """
    os.makedirs(outdir, exist_ok=True)
    if random_seed is not None:
        seed(random_seed)

    analysis_data = AnalysisData.generate_data(
        *true_params,
        gap_range=gap_range,
        Nf=Nf, tmax=tmax,
        alpha=alpha,
        filter=filter,
        noise=noise_realisation,
        plotfn=f"{outdir}/data.png",
    )

    x0 = sample_prior(PRIOR, nwalkers)  # Starting coordinates
    nwalkers, ndim = x0.shape

    # Check likelihood
    llike_val = log_posterior(true_params, analysis_data)
    print("Value of likelihood at true values is", llike_val)
    if noise_realisation is False:
        assert np.isclose(llike_val, 0.0), "Likelihood at true values is not zero!"

    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, pool=pool,
        args=(analysis_data,),
    )
    sampler.run_mcmc(x0, n_iter, progress=True)
    pool.close()

    # Save the chain
    idata_fname = os.path.join(outdir, "emcee_chain.nc")
    idata = az.from_emcee(sampler, var_names=["a", "ln_f", "ln_fdot"])
    idata = az.InferenceData(
        posterior=idata.posterior,
        sample_stats=idata.sample_stats,
    )
    # TODO: can i save true values here + real data?

    idata.to_netcdf(idata_fname)
    print(f"Saved chain to {idata_fname}")

    print("Making plots")
    plot_corner(idata_fname, trues=true_params, fname=f'{outdir}/corner.png')
    plot_mcmc_summary(
        idata_fname, analysis_data,
        fname=f'{outdir}/summary.png'
    )
