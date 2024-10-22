import emcee
from wavelet_domain_noise_with_gaps_nan import lnl, gap_hwavelet_generator, generate_data, A_TRUE, F_TRUE, FDOT_TRUE, \
    START_GAP, END_GAP, NF, TMAX, ONE_HOUR, OUTDIR
from scipy.stats import uniform
import numpy as np
import corner

A_RANGE = [1e-23, 1e-19]
F_RANGE = [1e-4, 1e-2]
FDOT_RANGE = [1e-12, 1e-6]

PRIOR = dict(
    a=uniform(A_RANGE[0], A_RANGE[1] - A_RANGE[0]),
    f=uniform(F_RANGE[0], F_RANGE[1] - F_RANGE[0]),
    fdot=uniform(FDOT_RANGE[0], FDOT_RANGE[1] - FDOT_RANGE[0])
)


def log_prior(theta):
    a, f, fdot = theta
    sample = dict(a=a, f=f, fdot=fdot)
    for key, dist in PRIOR.items():
        if not (dist.ppf(0) <= sample[key] <= dist.ppf(1)):
            return -np.inf
    return 0.0


def sample_prior(n_samples=1):
    """Return (nsamp, ndim) array of samples"""
    return np.array([
        [dist.rvs() for dist in PRIOR.values()]
        for _ in range(n_samples)
    ])


def log_probability(theta, gap, Nf, data, psd):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnl(*theta, gap, Nf, data, psd)


def trace_plot(sampler, trues=[A_TRUE, F_TRUE, FDOT_TRUE]):
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["a", "f", "fdot"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    plt.savefig(f"{OUTDIR}/trace_plot.pdf", bbox_inches="tight")


def plot_corner(sampler, discard=1000, thin=15, trues=[A_TRUE, F_TRUE, FDOT_TRUE]):
    flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
    labels = ["a", "f", "fdot"]
    fig = corner.corner(
        flat_samples, labels=labels, truths=trues
    )
    plt.savefig(f"{OUTDIR}/corner_plot.pdf", bbox_inches="tight")


def main(
        a_true=A_TRUE,
        f_true=F_TRUE,
        fdot_true=FDOT_TRUE,
        start_gap=START_GAP,
        end_gap=END_GAP,
        Nf=NF,
        tmax=TMAX,
        n_iter=1000,
        nwalkers=12
):
    data, psd, gap = generate_data(a_true, f_true, fdot_true, start_gap, end_gap, Nf, tmax)

    x0 = sample_prior(nwalkers)
    nwalkers, ndim = x0.shape

    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_probability, args=(gap, Nf, data, psd)
    )
    sampler.run_mcmc(x0, n_iter, progress=True)
    trace_plot(sampler)

    tau = sampler.get_autocorr_time(quiet=True)
    print("Autocorrelation time:", tau)


    plot_corner(sampler)


if __name__ == "__main__":
    main()
