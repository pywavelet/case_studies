import emcee
from typing import List
from bilby.core.prior import Uniform, TruncatedGaussian, Gaussian, PriorDict
import time

from multiprocessing import (get_context, cpu_count)
import os
import arviz as az
import warnings

from .plotting import plot_corner, make_mcmc_trace_gif, plot_mcmc_summary
from .random import seed
from .constants import *
from .analysis_data import AnalysisData
from .gap_window import GapType
from .signal_utils import waveform


PRIOR = PriorDict(dict(
    a=Uniform(*A_RANGE),
    ln_f=Uniform(*LN_F_RANGE),
    ln_fdot=Uniform(*LN_FDOT_RANGE)
))

CENTERED_PRIOR = PriorDict(dict(
    a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE * 0.1, minimum=1e-25, maximum=1e-15),
    ln_f=Gaussian(mu=LN_F_TRUE, sigma=LN_F_SCALE * 0.1),
    ln_fdot=Gaussian(mu=LN_FDOT_TRUE, sigma=LN_FDOT_SCALE * 0.1)
))


def log_prior(theta):
    a, ln_f, ln_fdot = theta
    _lnp = PRIOR.ln_prob(dict(a=a, ln_f=ln_f, ln_fdot=ln_fdot))
    if not np.isfinite(_lnp):
        return -np.inf
    else:
        return 0.0


def sample_prior(prior: PriorDict, n_samples=1) -> np.ndarray:
    """Return (nsamp, ndim) array of samples"""
    return np.array(list(prior.sample(n_samples).values())).T


def log_posterior(theta: List[float], analysis_data:AnalysisData) -> float:
    _lp = log_prior(theta)
    if not np.isfinite(_lp):
        return -np.inf
    else:
        _lnl = analysis_data.lnl(*theta)
        return _lp + _lnl


def run_mcmc(
        true_params=[A_TRUE, LN_F_TRUE, LN_FDOT_TRUE],
        gap_ranges=GAP_RANGES,
        Nf=NF,
        tmax=TMAX,
        dt=DT,
        alpha=0.0,
        highpass_fmin=10 ** -4,
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
    _start_time = time.time()
    os.makedirs(outdir, exist_ok=True)
    if random_seed is not None:
        seed(random_seed)

    analysis_data = AnalysisData(
        data_kwargs=dict(
            dt=dt, noise=noise_realisation, tmax=tmax,
            highpass_fmin=highpass_fmin,
            alpha=alpha,
            Nf=Nf
        ),
        gap_kwargs=dict(
            type=GapType.STITCH,
            gap_ranges=gap_ranges
        ),
        waveform_generator=waveform,
        waveform_parameters=true_params,
        plotfn=f"{outdir}/data.png",
    )

    x0 = sample_prior(PRIOR, nwalkers)  # Starting coordinates
    nwalkers, ndim = x0.shape

    # Check likelihood
    llike_val = log_posterior(true_params, analysis_data)
    print("Value of likelihood at true values is", llike_val)
    if noise_realisation is False and not np.isclose(llike_val, 0.0):
        warnings.warn(f"LnL(True values) = {llike_val:.3e} != 0.0... SUSPICIOUS!")

    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_posterior, pool=pool,
        args = (analysis_data,)
    )
    sampler.run_mcmc(x0, n_iter, progress=True)
    pool.close()

    runtime = time.time() - _start_time

    # Save the chain
    idata_fname = os.path.join(outdir, "emcee_chain.nc")
    idata = az.from_emcee(sampler, var_names=["a", "ln_f", "ln_fdot"])
    idata.sample_stats["runtime"] = runtime
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
    print(f"Runtime: {_fmt_rutime(float(idata.sample_stats.runtime))}")

def _fmt_rutime(t:float):
    hours, remainder = divmod(t, 3600)
    minutes, seconds = divmod(remainder, 60)
    fmt = ""
    if hours:
        fmt += f"{int(hours)}h"
    if minutes:
        fmt += f"{int(minutes)}m"
    if seconds:
        fmt += f"{int(seconds)}s"
    return fmt
