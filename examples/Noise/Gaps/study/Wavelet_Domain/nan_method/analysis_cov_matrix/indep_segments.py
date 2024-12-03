import sys

import emcee

sys.path.append("..")
import os
from multiprocessing import cpu_count, get_context

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
from constants import (
    A_TRUE,
    CENTERED_PRIOR,
    END_GAP,
    LN_F_TRUE,
    LN_FDOT_TRUE,
    NF,
    ONE_HOUR,
    OUTDIR,
    PRIOR,
    START_GAP,
    TMAX,
    TRUES,
)
from pywavelet.transforms import from_freq_to_wavelet, from_time_to_wavelet
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from tqdm import tqdm as tqdm
from wavelet_domain_noise_with_gaps_nan import (
    generate_data,
    generate_padded_signal,
    generate_stat_noise,
    generate_wavelet_with_gap,
)

from gap_study_utils.noise_curves import (
    CornishPowerSpectralDensity,
    noise_PSD_AE,
)
from gap_study_utils.wavelet_data_utils import chunk_timeseries


def main(
    a_true=A_TRUE,
    ln_f_true=LN_F_TRUE,
    ln_fdot_true=LN_FDOT_TRUE,
    start_gap=START_GAP,
    end_gap=END_GAP,
    Nf=NF,
    tmax=TMAX,
):

    windowing, alpha = (
        False,
        0.0,
    )  # Set this parameter if you want to window (reduce leakage)
    filter = True  # Set this parameter if you wish to apply a high pass filter
    noise_realisation = True
    ht, hf = generate_padded_signal(a_true, ln_f_true, ln_fdot_true, tmax)

    psd = FrequencySeries(
        data=CornishPowerSpectralDensity(hf.freq), freq=hf.freq
    )

    _, _, gap = generate_data(
        a_true,
        ln_f_true,
        ln_fdot_true,
        start_gap,
        end_gap,
        Nf,
        tmax,
        windowing=windowing,
        alpha=alpha,
        filter=filter,
        noise_realisation=noise_realisation,
        seed_no=11_07_1993,
    )

    flattened_vec_no_gap = []
    flattened_vec_gap_seg_0 = []
    flattened_vec_gap_seg_1 = []
    for i in tqdm(range(0, 10000)):

        noise_FS = generate_stat_noise(ht, psd, seed_no=i, TD=False)
        noise_TS = generate_stat_noise(ht, psd, seed_no=i, TD=True)

        noise_wavelet = from_freq_to_wavelet(noise_FS, Nf=Nf)

        # Split data stream into chunks
        Chunked_Data = chunk_timeseries(noise_TS, gap)

        noise_wavelet_individual_chunks = [
            from_time_to_wavelet(item, Nf=Nf) for item in Chunked_Data
        ]

        flattened_vec_no_gap.append(noise_wavelet.data.flatten())
        flattened_vec_gap_seg_0.append(
            noise_wavelet_individual_chunks[0].data.flatten()
        )
        flattened_vec_gap_seg_1.append(
            noise_wavelet_individual_chunks[-1].data.flatten()
        )

    Cov_Matrix_no_gap = np.cov(flattened_vec_no_gap, rowvar=False)
    Cov_Matrix_seg_0 = np.cov(flattened_vec_gap_seg_0, rowvar=False)
    Cov_Matrix_seg_1 = np.cov(flattened_vec_gap_seg_1, rowvar=False)

    # This is in the git ignore
    np.save("matrix_directory/Cov_Matrix_Flat_w_filter.npy", Cov_Matrix_no_gap)
    np.save("matrix_directory/Cov_Matrix_Flat_w_seg_0.npy", Cov_Matrix_seg_0)
    np.save("matrix_directory/Cov_Matrix_Flat_w_seg_0.npy", Cov_Matrix_seg_1)


if __name__ == "__main__":
    main()
