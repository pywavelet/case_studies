from wavelet_domain_noise_with_gaps_nan import lnl, gap_hwavelet_generator, generate_data, A_TRUE, F_TRUE, FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, ONE_HOUR, OUTDIR


def main(
    a_true = A_TRUE,
    f_true = F_TRUE,
    fdot_true = FDOT_TRUE,
    start_gap = START_GAP,
    end_gap = END_GAP,
    Nf = NF,
    tmax = TMAX,
):

    hwavelet_with_gap, psd_wavelet_with_gap, gap = generate_data(a_true, f_true, fdot_true, start_gap, end_gap, Nf, tmax)

    precision = a_true / np.sqrt(np.nansum(h_wavelet.data ** 2 / psd_wavelet.data))
    a_range = np.linspace(a_true - 5 * precision, a_true + 5 * precision, 200)
    lnl_kwgs = dict(data=hwavelet_with_gap, psd=psd_wavelet_with_gap, gap=gap, Nf=Nf)
    llike_vec = np.array([lnl(_a, f_true, fdot_true, **lnl_kwgs) for _a in tqdm(a_range)])


    # Convert to array and plot
    plt.figure()
    plt.plot(a_range, np.exp(llike_vec))
    plt.axvline(x=a_true, c='red', linestyle='--', label="truth")
    plt.legend()
    plt.xlabel(r'Amplitude values')
    plt.ylabel(r'Likelihood')
    plt.title(f'Likelihood with Nf = {Nf}')
    plt.savefig(os.path.join(OUTDIR, f"likelihood_Nf_{Nf}.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
