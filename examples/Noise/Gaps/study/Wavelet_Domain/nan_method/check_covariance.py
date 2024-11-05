import os

from gwpy.signal.filter_design import highpass

from gap_study_utils.analysis_data  import AnalysisData, generate_wavelet_with_gap

from gap_study_utils.constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, START_GAP, END_GAP, NF, TMAX
from pywavelet.transforms import from_freq_to_wavelet, from_time_to_wavelet
from gap_study_utils.random import seed
import numpy as np
from tqdm.auto import trange
from gap_study_utils.noise_curves import generate_stationary_noise

def main(
        a_true=A_TRUE,
        ln_f_true=LN_F_TRUE,
        ln_fdot_true=LN_FDOT_TRUE,
        start_gap=START_GAP,
        end_gap=END_GAP,
        Nf=NF,
        tmax=TMAX,
        N_iter=10_000,
        outdir="matrix_directory"
):
    os.makedirs(outdir, exist_ok=True)
    seed(11_07_1993)
    analysis_data = AnalysisData.generate_data(
        a_true=a_true,
        ln_f_true=ln_f_true,
        ln_fdot_true=ln_fdot_true,
        gap_range=[start_gap,end_gap],
        Nf=Nf,
        tmax=tmax,
        noise=True,
        alpha=0.0,
        highpass_fmin=0
    )


    flattened_vec_gap = []
    flattened_vec_no_gap = []
    for i in trange(N_iter):
        seed(i)
        noise_TS = generate_stationary_noise(ND=analysis_data.ND, dt=analysis_data.dt, psd=analysis_data.psd_freqseries, time_domain=True)
        noise_FS = generate_stationary_noise(ND=analysis_data.ND, dt=analysis_data.dt, psd=analysis_data.psd_freqseries, time_domain=False)
        noise_wavelet = from_freq_to_wavelet(noise_FS, Nf = Nf)
        noise_wavelet_with_gap = generate_wavelet_with_gap(
            gap=analysis_data.gap,
            ht=noise_TS,
            Nf=analysis_data.Nf,
            alpha=analysis_data.alpha,
            highpass_fmin=analysis_data.highpass_fmin
        )

        noise_wavelet_flat = noise_wavelet.data.flatten()
        noise_wavelet_with_gap_flat = noise_wavelet_with_gap.data.flatten()
        noise_wavelet_with_gap_flat = np.nan_to_num(noise_wavelet_with_gap_flat, nan=0)
        

        flattened_vec_no_gap.append(noise_wavelet_flat)
        flattened_vec_gap.append(noise_wavelet_with_gap_flat)

    Cov_Matrix_no_gap = np.cov(flattened_vec_no_gap, rowvar = False)
    Cov_Matrix_gap = np.cov(flattened_vec_gap, rowvar = False)
    # This is in the git ignore
    np.save(f"{outdir}/Cov_Matrix_Flat_w_filter.npy", Cov_Matrix_no_gap)
    np.save(f"{outdir}/Cov_Matrix_Flat_w_filter_gap.npy", Cov_Matrix_gap)

if __name__ == "__main__":
    main(N_iter=10_000)
