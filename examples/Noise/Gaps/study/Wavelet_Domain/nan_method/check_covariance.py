import emcee
from wavelet_domain_noise_with_gaps_nan import generate_data, generate_stat_noise, generate_wavelet_with_gap, generate_padded_signal
from constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, START_GAP, END_GAP, NF, TMAX, ONE_HOUR, OUTDIR, PRIOR, TRUES, CENTERED_PRIOR
from multiprocessing import (get_context,cpu_count)
from gap_study_utils.noise_curves import noise_PSD_AE, CornishPowerSpectralDensity
from pywavelet.utils import evolutionary_psd_from_stationary_psd, compute_snr, compute_likelihood
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.transforms import from_freq_to_wavelet, from_time_to_wavelet
import numpy as np
from tqdm import tqdm as tqdm  
import os
import arviz as az

import matplotlib.pyplot as plt

def main(
        a_true=A_TRUE,
        ln_f_true=LN_F_TRUE,
        ln_fdot_true=LN_FDOT_TRUE,
        start_gap=START_GAP,
        end_gap=END_GAP,
        Nf=NF,
        tmax=TMAX,
):
    
    windowing, alpha = False, 0.0 # Set this parameter if you want to window (reduce leakage)
    filter = True # Set this parameter if you wish to apply a high pass filter
    noise_realisation = True
    ht, hf = generate_padded_signal(a_true, ln_f_true, ln_fdot_true, tmax)
    h_wavelet = from_freq_to_wavelet(hf, Nf=Nf)
    
    psd = FrequencySeries(
        data=CornishPowerSpectralDensity(hf.freq),
        freq=hf.freq
    )

    _, _, gap = generate_data(a_true, ln_f_true, ln_fdot_true, 
                                    start_gap, end_gap, Nf, tmax, 
                                    windowing = windowing,
                                    alpha = alpha,
                                    filter = filter,
                                    noise_realisation = noise_realisation,
                                    seed_no = 11_07_1993)


    flattened_vec_gap = []
    flattened_vec_no_gap = []
    for i in tqdm(range(0,10000)):

        noise_FS = generate_stat_noise(ht, psd, seed_no = i, TD = False)
        noise_TS = generate_stat_noise(ht, psd, seed_no = i, TD = True)

        noise_wavelet = from_freq_to_wavelet(noise_FS, Nf = Nf)

        noise_wavelet_with_gap = generate_wavelet_with_gap(gap, noise_TS, Nf, 
                                                           windowing=windowing, alpha=alpha,
                                                          filter=filter)

        noise_wavelet_flat = noise_wavelet.data.flatten()
        noise_wavelet_with_gap_flat = noise_wavelet_with_gap.data.flatten()
        noise_wavelet_with_gap_flat = np.nan_to_num(noise_wavelet_with_gap_flat, nan=0)
        

        flattened_vec_no_gap.append(noise_wavelet_flat)
        flattened_vec_gap.append(noise_wavelet_with_gap_flat)

    Cov_Matrix_no_gap = np.cov(flattened_vec_no_gap, rowvar = False)
    Cov_Matrix_gap = np.cov(flattened_vec_gap, rowvar = False)
    # This is in the git ignore
    np.save("matrix_directory/Cov_Matrix_Flat_w_filter.npy", Cov_Matrix_no_gap)
    np.save("matrix_directory/Cov_Matrix_Flat_w_filter_gap.npy", Cov_Matrix_gap)

if __name__ == "__main__":
    main()
