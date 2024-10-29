from wavelet_domain_noise_with_gaps_nan import lnl, gap_hwavelet_generator, generate_data
from constants import A_TRUE, F_TRUE, FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, ONE_HOUR, OUTDIR, F_RANGE, FDOT_RANGE
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import gif

gif.options.matplotlib["dpi"] = 100


def plot_lnl(params, lnl_vec, param_name, true_value, ax=None, curr_val=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(params, lnl_vec)
    ax.axvline(x=true_value, c='red', linestyle='--', label="truth")
    if curr_val is not None:
        ax.axvline(x=curr_val, c='green', linestyle='--', label="current")
    ax.set_xlabel(param_name)
    ax.set_ylabel(r'LnL')
    ax.legend()


def plot_wavelets(hdata, htemplate, ax=None):
    if ax is None:
        fig, ax = plt.subplots(2, 1)
    hdata.plot(ax=ax[0])
    htemplate.plot(ax=ax[1])


@gif.frame
def plot(x, lnl_kwargs, waveform_kwarg, hdata):
    lnl_kwargs['curr_val'] = x
    waveform_kwarg[lnl_kwargs['param_name']] = x
    htemplate = gap_hwavelet_generator(**waveform_kwarg)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    plot_lnl(ax=axes[0], **lnl_kwargs)
    plot_wavelets(hdata, htemplate, ax=[axes[1], axes[2]])


def make_gif(xparams, lnl_kwargs, waveform_kwarg, hdata):
    frames = []
    param_label = lnl_kwargs["param_name"]
    for i, x in enumerate(tqdm(xparams, desc=f"Generating {param_label} frames")):
        frames.append(
            plot(x, lnl_kwargs, waveform_kwarg, hdata)
        )
    gif.save(frames, f"{OUTDIR}/{param_label}_lnl.gif", duration=100)


def main(
        a_true=A_TRUE,
        f_true=F_TRUE,
        fdot_true=FDOT_TRUE,
        start_gap=START_GAP,
        end_gap=END_GAP,
        Nf=NF,
        tmax=TMAX,
        N_points=15
):
    hwavelet, psd, gap = generate_data(a_true, f_true, fdot_true, start_gap, end_gap, Nf, tmax)

    precision = a_true / np.sqrt(np.nansum(hwavelet.data ** 2 / psd.data))
    a_range = np.linspace(a_true - 5 * precision, a_true + 5 * precision, N_points)
    f_range = np.linspace(*F_RANGE, N_points)
    fdot_range = np.linspace(*FDOT_RANGE, N_points)
    kwgs = dict(Nf=Nf, gap=gap)
    lnl_kwgs = dict(**kwgs, psd=psd, data=hwavelet)
    wfm_kwgs = dict(a=a_true, f=f_true, fdot=fdot_true, **kwgs)


    a_lnls_vec = np.array([lnl(_a, f_true, fdot_true, **lnl_kwgs) for _a in tqdm(a_range)])
    make_gif(
        a_range,
        dict(param_name='a', lnl_vec=a_lnls_vec, params=a_range, true_value=a_true),
        wfm_kwgs,
        hwavelet,
    )

    f_lnls_vec = np.array([lnl(a_true, _f, fdot_true,  **lnl_kwgs) for _f in tqdm(f_range)])
    make_gif(
        f_range,
        dict(param_name='f', lnl_vec=f_lnls_vec, params=f_range, true_value=f_true),
        wfm_kwgs,
        hwavelet,
    )

    fdot_lnls_vec = np.array([lnl(a_true,f_true, _fdot, **lnl_kwgs) for _fdot in tqdm(fdot_range)])
    make_gif(
        fdot_range,
        dict(param_name='fdot', lnl_vec=fdot_lnls_vec, params=fdot_range, true_value=fdot_true),
        wfm_kwgs,
        hwavelet,
    )




if __name__ == "__main__":
    main()
