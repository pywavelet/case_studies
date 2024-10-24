import emcee
from wavelet_domain_noise_with_gaps_nan import lnl, gap_hwavelet_generator, generate_data
from constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, ONE_HOUR, OUTDIR, PRIOR, TRUES, CENTERED_PRIOR
from scipy.stats import uniform
import numpy as np
import corner
import os
import arviz as az

import matplotlib.pyplot as plt


OUT_MCMC = os.path.join(OUTDIR, "mcmc")
os.makedirs(OUT_MCMC, exist_ok=True)



def log_prior(theta):
    a, ln_f, ln_fdot = theta
    return PRIOR.ln_prob(dict(a=a, ln_f=ln_f, ln_fdot=ln_fdot))


def sample_prior(prior, n_samples=1):
    """Return (nsamp, ndim) array of samples"""
    return np.array(list(prior.sample(n_samples).values())).T


def log_probability(theta, gap, Nf, data, psd):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnl(*theta, gap, Nf, data, psd)


def plot_prior():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (key, dist) in zip(axes, PRIOR.items()):
        x = np.sort(dist.sample(1000))
        ax.plot(x, dist.prob(x), label="prior")
        ax.set_title(key)
    for t in TRUES:
        ax.axvline(t, c="red", linestyle="--", label="truth")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_MCMC, "prior.png"))


def main(
        a_true=A_TRUE,
        ln_f_true=LN_F_TRUE,
        ln_fdot_true=LN_FDOT_TRUE,
        start_gap=START_GAP,
        end_gap=END_GAP,
        Nf=NF,
        tmax=TMAX,
        n_iter=25000,
        nwalkers=32
):
    plot_prior()
    data, psd, gap = generate_data(a_true, ln_f_true, ln_fdot_true, start_gap, end_gap, Nf, tmax)


    x0 = sample_prior(CENTERED_PRIOR, nwalkers)
    nwalkers, ndim = x0.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(gap, Nf, data, psd)
    )
    sampler.run_mcmc(x0, n_iter, progress=True)

    # Save the chain
    idata = az.from_emcee(sampler, var_names=["a", "ln_f", "ln_fdot"])
    idata = az.InferenceData(posterior=idata.posterior, sample_stats=idata.sample_stats)
    idata.to_netcdf("emcee_chain.nc")
    print("Saved chain to emcee_chain.nc")




if __name__ == "__main__":
    main()
