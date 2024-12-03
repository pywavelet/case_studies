import arviz as az
import corner
import gif
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import trange

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import TRUES

gif.options.matplotlib["dpi"] = 100


def plot_trace(idata: az.InferenceData, axes, i=None, max_iter=None):
    if i is not None:
        sliced_posterior = idata.posterior.isel(
            chain=slice(None), draw=slice(0, i)
        )
        idata = az.InferenceData(posterior=sliced_posterior)

    az.plot_trace(idata, axes=axes)
    for row in range(3):
        axes[row, 0].axvline(
            TRUES[row], c="red", linestyle="--", label="truth"
        )
        axes[row, 1].axhline(
            TRUES[row], c="red", linestyle="--", label="truth"
        )
        if i is not None:
            axes[row, 1].axvline(i, c="green", linestyle="--")
            axes[row, 1].set_xlim(0, max_iter)
    axes[0, 0].set_xscale("log")
    axes[0, 1].set_yscale("log")


@gif.frame
def _trace_mcmc_frame(idata, analysis_data: AnalysisData, i, max_iter=None):
    plot_mcmc_summary(idata, analysis_data, max_iter)


def plot_mcmc_summary(idata, analysis_data: AnalysisData, i=None, fname=None):
    if isinstance(idata, str):
        idata = az.from_netcdf(idata)

    max_iter = len(idata.sample_stats.draw)
    i = i or len(idata.sample_stats.draw) - 1
    ith_samp = idata.posterior.isel(draw=i).median(dim="chain")
    ith_samp = {
        param: float(ith_samp[param].values) for param in ith_samp.data_vars
    }
    htemplate = analysis_data.htemplate(**ith_samp)

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    fig.suptitle(f"Iteration {i}")
    plot_trace(idata, axes, i, max_iter)
    analysis_data.data_wavelet.plot(
        ax=axes[3, 0],
        show_colorbar=False,
        label="Whiten Data\n",
        whiten_by=analysis_data.psd.data,
        absolute=True,
        zscale="log",
    )
    htemplate.plot(
        ax=axes[3, 1],
        show_colorbar=False,
        label="ith-sample Signal\n",
        absolute=True,
        zscale="log",
    )
    if fname:
        plt.tight_layout()
        plt.savefig(fname)


def make_mcmc_trace_gif(
    idata_fname, analysis_data, n_frames=20, fname="mcmc.gif"
):
    idata = az.from_netcdf(idata_fname)
    trace_frames = []
    N = len(idata.sample_stats.draw)
    trace_frames.append(_trace_mcmc_frame(idata, analysis_data, 1, N))
    for i in trange(int(N * 0.1), N, int(N / n_frames)):
        trace_frames.append(_trace_mcmc_frame(idata, analysis_data, i, N))
    gif.save(trace_frames, fname, duration=100)


def plot_corner(idata_fname, trues=TRUES, fname="corner.png"):
    idata = az.from_netcdf(idata_fname)
    # discard burn-in
    burnin = 0.5
    burnin_idx = int(burnin * len(idata.sample_stats.draw))
    idata = idata.sel(draw=slice(burnin_idx, None))
    idata.posterior["ln_f"] = np.exp(idata.posterior.ln_f)
    idata.posterior["ln_fdot"] = np.exp(idata.posterior.ln_fdot)
    trues = trues.copy()
    for i in range(1, 3):
        trues[i] = np.exp(trues[i])

    corner.corner(
        idata, truths=trues, labels=["a", "f", "fdot"], axes_scale="log"
    )
    plt.savefig(fname)


def plot_prior(prior: "bilby.core.prior.PriorDict", fname="prior.png"):
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    for ax, (key, dist) in zip(axes, prior.items()):
        x = np.sort(dist.sample(1000))
        ax.plot(x, dist.prob(x), label="prior")
        ax.set_title(key)
    for i, t in enumerate(TRUES):
        axes[i].axvline(t, c="red", linestyle="--", label="truth")
    plt.tight_layout()
    plt.savefig(fname)
