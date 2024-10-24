import gif
import matplotlib.pyplot as plt
import arviz as az
from constants import TRUES, PRIOR, A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, RANGES
from wavelet_domain_noise_with_gaps_nan import lnl, gap_hwavelet_generator, generate_data
from tqdm.auto import trange
import corner
import numpy as np

FNAME = 'emcee_chain.nc'
gif.options.matplotlib["dpi"] = 100

IDATA = az.from_netcdf(FNAME)

def plot_trace(idata:az.InferenceData, axes, i=None, max_iter=None):
    if i is not None:
        sliced_posterior =  idata.posterior.isel(chain=slice(None), draw=slice(0, i))
        idata = az.InferenceData(posterior=sliced_posterior)

    az.plot_trace(idata, axes=axes)
    for row in range(3):
        axes[row, 0].axvline(TRUES[row], c='red', linestyle='--', label='truth')
        axes[row, 1].axhline(TRUES[row], c='red', linestyle='--', label='truth')
        if i is not None:
            axes[row, 1].axvline(i, c='green', linestyle='--')
            axes[row, 1].set_xlim(0, max_iter)
        # # set lims
        # axes[row, 0].set_xlim(*RANGES[row])
        # axes[row, 1].set_ylim(*RANGES[row])
    axes[0,0].set_xscale('log')
    axes[0,1].set_yscale('log')

def _get_samp(i):
    samp = IDATA.posterior.isel(draw=i).median(dim="chain")
    return  {param: float(samp[param].values) for param in samp.data_vars}


@gif.frame
def frame(hdata, kwgs, i, max_iter=None):
    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    fig.suptitle(f"Iteration {i}")
    plot_trace(IDATA, axes, i, max_iter)
    hdata.plot(ax=axes[3,0], show_colorbar=False)
    htemplate = gap_hwavelet_generator(**_get_samp(i), **kwgs)
    htemplate.plot(ax=axes[3,1], show_colorbar=False)

def make_gif(hdata, kwgs, n_frames = 20):
    frames = []
    N = len(IDATA.sample_stats.draw)
    for i in trange(int(N*0.1), N, int(N/n_frames)):
        frames.append(frame(hdata, kwgs, i, N))
    gif.save(frames, "mcmc.gif", duration=100)

def plot_corner():
    idata = az.from_netcdf(FNAME)
    # discard burn-in
    burnin = 0.5
    burnin_idx = int(burnin * len(idata.sample_stats.draw))
    idata = idata.sel(draw=slice(burnin_idx, None))
    # change a to log_a
    idata.posterior['a'] = np.log(idata.posterior.a)
    trues = TRUES.copy()
    trues[0] = np.log(trues[0])
    ranges = RANGES.copy()
    ranges[0] = np.log(ranges[0])
    corner.corner(idata, truths=trues, labels=["log_a", "ln_f", "ln_fdot"])
    plt.savefig("corner.png")

def main(
        a_true=A_TRUE,
        ln_f_true=LN_F_TRUE,
        ln_fdot_true=LN_FDOT_TRUE,
        start_gap=START_GAP,
        end_gap=END_GAP,
        Nf=NF,
        tmax=TMAX,
):
    plot_corner()
    # plot_trace()
    hdata, psd, gap = generate_data(a_true, ln_f_true, ln_fdot_true, start_gap, end_gap, Nf, tmax)
    kwgs = dict(gap=gap, Nf=Nf)
    make_gif(hdata, kwgs)


if __name__ == '__main__':
    main()