import os

import numpy as np
from pywavelet.transforms import from_freq_to_wavelet, from_time_to_wavelet
from tqdm.auto import trange

from gap_study_utils.analysis_data import AnalysisData
from gap_study_utils.constants import (
    A_TRUE,
    GAP_RANGES,
    LN_F_TRUE,
    LN_FDOT_TRUE,
    NF,
    TMAX,
)
from gap_study_utils.noise_curves import generate_stationary_noise
from gap_study_utils.random import seed


def main(N_iter=10_000, outdir="matrix_directory", data_kwargs={}):
    os.makedirs(outdir, exist_ok=True)
    seed(11_07_1993)

    flattened_vec_gap = []
    flattened_vec_no_gap = []
    for i in trange(N_iter):
        seed(i)
        data = AnalysisData.generate_data(**data_kwargs)

        noise_TS = generate_stationary_noise(
            ND=analysis_data.ND,
            dt=analysis_data.dt,
            psd=analysis_data.psd_freqseries,
            time_domain=True,
        )
        noise_FS = generate_stationary_noise(
            ND=analysis_data.ND,
            dt=analysis_data.dt,
            psd=analysis_data.psd_freqseries,
            time_domain=False,
        )
        noise_wavelet = noise_FS.to_wavelet(Nf=analysis_data.Nf)
        noise_wavelet_with_gap = gap

        noise_wavelet_flat = noise_wavelet.data.flatten()
        noise_wavelet_with_gap_flat = noise_wavelet_with_gap.data.flatten()

        flattened_vec_no_gap.append(noise_wavelet_flat)
        flattened_vec_gap.append(noise_wavelet_with_gap_flat)

    Cov_Matrix_no_gap = np.cov(flattened_vec_no_gap, rowvar=False)
    Cov_Matrix_gap = np.cov(flattened_vec_gap, rowvar=False)
    # This is in the git ignore
    np.save(f"{outdir}/Cov_Matrix_Flat_w_filter.npy", Cov_Matrix_no_gap)
    np.save(f"{outdir}/Cov_Matrix_Flat_w_filter_gap.npy", Cov_Matrix_gap)


if __name__ == "__main__":
    main(N_iter=10_000)
