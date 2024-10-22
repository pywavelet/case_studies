from wavelet_domain_noise_with_gaps_nan import lnl, gap_hwavelet_generator, generate_data, A_TRUE, F_TRUE, FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, ONE_HOUR, OUTDIR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def plot_lnl(params, lnl_vec, param_name, true_value, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(params, np.exp(lnl_vec))
    ax.axvline(x=true_value, c='red', linestyle='--', label="truth")
    ax.set_xlabel(param_name)
    ax.set_ylabel(r'Likelihood')




def main(
    a_true = A_TRUE,
    f_true = F_TRUE,
    fdot_true = FDOT_TRUE,
    start_gap = START_GAP,
    end_gap = END_GAP,
    Nf = NF,
    tmax = TMAX,
):

    hwavelet, psd, gap = generate_data(a_true, f_true, fdot_true, start_gap, end_gap, Nf, tmax)

    precision = a_true / np.sqrt(np.nansum(hwavelet.data ** 2 / psd.data))
    a_range = np.linspace(a_true - 5 * precision, a_true + 5 * precision, 200)
    f_range = np.geomspace(hwavelet.freq[1], hwavelet.freq[-1], 200)

    lnl_kwgs = dict(data=hwavelet, psd=psd, gap=gap, Nf=Nf)
    a_lnls_vec = np.array([lnl(_a, f_true, fdot_true, **lnl_kwgs) for _a in tqdm(a_range)])
    f_lnls_vec = np.array([lnl(a_true, _f, fdot_true, **lnl_kwgs) for _f in tqdm(f_range)])


    # Convert to array and plot
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    plot_lnl(a_range, a_lnls_vec, "a", a_true, ax=ax[0])
    plot_lnl(f_range, f_lnls_vec, "f", f_true, ax=ax[1])
    ax[1].set_xscale("log")
    fig.savefig(f"{OUTDIR}/lnl.pdf", bbox_inches="tight")

    hwavelet, psd, gap = generate_data(a_true, f_range[10], fdot_true, start_gap, end_gap, Nf, tmax, plotfn="best_f.pdf")


if __name__ == "__main__":
    main()
