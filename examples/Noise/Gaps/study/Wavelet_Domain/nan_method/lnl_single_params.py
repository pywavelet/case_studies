import os

import gif
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from gap_study_utils.analysis_data import generate_data
from gap_study_utils.bayesian_functions import gap_hwavelet_generator, lnl
from gap_study_utils.constants import (
    A_TRUE,
    END_GAP,
    LN_F_RANGE,
    LN_F_TRUE,
    LN_FDOT_RANGE,
    LN_FDOT_TRUE,
    NF,
    ONE_HOUR,
    START_GAP,
    TMAX,
)

OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)

gif.options.matplotlib["dpi"] = 100


def plot_lnl(params, lnl_vec, param_name, true_value, ax=None, curr_val=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(params, lnl_vec)
    ax.axvline(x=true_value, c="red", linestyle="--", label="truth")
    if curr_val is not None:
        ax.axvline(x=curr_val, c="green", linestyle="--", label="current")
    ax.set_xlabel(param_name)
    ax.set_ylabel(r"LnL")
    ax.legend()


def plot_wavelets(hdata, htemplate, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2, 1)
    hdata.plot(ax=ax[0])
    htemplate.plot(ax=ax[1])
    htemplate.plot_trend(ax=ax[0], color="black")


@gif.frame
def plot(x, lnl_kwargs, waveform_kwarg, hdata):
    lnl_kwargs["curr_val"] = x
    waveform_kwarg[lnl_kwargs["param_name"]] = x
    htemplate = gap_hwavelet_generator(**waveform_kwarg)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    plot_lnl(ax=axes[0], **lnl_kwargs)
    plot_wavelets(hdata, htemplate, ax=[axes[1], axes[2]])


def make_gif(xparams, lnl_kwargs, waveform_kwarg, hdata):
    frames = []
    param_label = lnl_kwargs["param_name"]
    for i, x in enumerate(
        tqdm(xparams, desc=f"Generating {param_label} frames")
    ):
        frames.append(plot(x, lnl_kwargs, waveform_kwarg, hdata))
    gif.save(frames, f"{OUTDIR}/{param_label}_lnl.gif", duration=100)


def main(
    a_true=A_TRUE,
    ln_f_true=LN_F_TRUE,
    ln_fdot_true=LN_FDOT_TRUE,
    start_gap=START_GAP,
    end_gap=END_GAP,
    Nf=NF,
    tmax=TMAX,
    N_points=10,
):
    hwavelet, psd, gap = generate_data(
        a_true, ln_f_true, ln_fdot_true, start_gap, end_gap, Nf, tmax
    )

    precision = a_true / np.sqrt(np.nansum(hwavelet.data**2 / psd.data))
    a_range = np.linspace(
        a_true - 5 * precision, a_true + 5 * precision, N_points
    )
    ln_f_range = np.linspace(*LN_F_RANGE, N_points)
    ln_fdot_range = np.linspace(*LN_FDOT_RANGE, N_points)
    kwgs = dict(Nf=Nf, gap=gap)
    lnl_kwgs = dict(**kwgs, psd=psd, data=hwavelet)
    wfm_kwgs = dict(a=a_true, ln_f=ln_f_true, ln_fdot=ln_fdot_true, **kwgs)

    a_lnls_vec = np.array(
        [lnl(_a, ln_f_true, ln_fdot_true, **lnl_kwgs) for _a in tqdm(a_range)]
    )
    make_gif(
        a_range,
        dict(
            param_name="a",
            lnl_vec=a_lnls_vec,
            params=a_range,
            true_value=a_true,
        ),
        wfm_kwgs,
        hwavelet,
    )

    f_lnls_vec = np.array(
        [
            lnl(a_true, _ln_f, ln_fdot_true, **lnl_kwgs)
            for _ln_f in tqdm(ln_f_range)
        ]
    )
    make_gif(
        ln_f_range,
        dict(
            param_name="ln_f",
            lnl_vec=f_lnls_vec,
            params=ln_f_range,
            true_value=ln_f_true,
        ),
        wfm_kwgs,
        hwavelet,
    )

    ln_fdot_lnls_vec = np.array(
        [
            lnl(a_true, ln_f_true, _ln_fdot, **lnl_kwgs)
            for _ln_fdot in tqdm(ln_fdot_range)
        ]
    )
    make_gif(
        ln_fdot_range,
        dict(
            param_name="ln_fdot",
            lnl_vec=ln_fdot_lnls_vec,
            params=ln_fdot_range,
            true_value=ln_fdot_true,
        ),
        wfm_kwgs,
        hwavelet,
    )


if __name__ == "__main__":
    main()
