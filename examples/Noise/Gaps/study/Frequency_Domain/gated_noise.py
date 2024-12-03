import os

import numpy as np
from pywavelet.transforms import from_freq_to_wavelet
from pywavelet.transforms.types import FrequencySeries, Wavelet
from pywavelet.utils import evolutionary_psd_from_stationary_psd
from scipy.signal.windows import tukey

from gap_study_utils.gap_funcs import gap_routine, get_Cov, regularise_matrix
from gap_study_utils.noise_curves import (
    CornishPowerSpectralDensity,
    noise_PSD_AE,
)
from gap_study_utils.signal_utils import inner_prod, waveform, zero_pad

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")


np.random.seed(1234)


def main(
    a_true=1e-21,
    f_true=3e-3,
    fdot_true=1e-8,
    TDI="Cornish",  # from TD1, TDI2, Cornish
    tmax=10 * 60 * 60,  # Final time
):
    # Set true parameters. These are the parameters we want to estimate using MCMC.

    fs = 2 * f_true  # Sampling rate
    delta_t = np.floor(
        0.4 / fs
    )  # Sampling interval -- largely oversampling here.

    t = np.arange(
        0, tmax, delta_t
    )  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

    N = int(
        2 ** (np.ceil(np.log2(len(t))))
    )  # Round length of time series to a power of two.
    # Length of time series
    h_t = waveform(a_true, f_true, fdot_true, t)
    taper_signal = tukey(len(h_t), alpha=0.2)
    h_t_pad = zero_pad(h_t * taper_signal)

    t_pad = np.arange(0, len(h_t_pad) * delta_t, delta_t)

    h_true_f = np.fft.rfft(h_t_pad)
    freq = np.fft.rfftfreq(N, delta_t)
    freq[0] = freq[1]
    if TDI == "TDI1" or TDI == "TDI2":
        PSD = noise_PSD_AE(freq, TDI=TDI)
    else:
        PSD = CornishPowerSpectralDensity(freq)
    SNR2 = inner_prod(
        h_true_f, h_true_f, PSD, delta_t, N
    )  # Compute optimal matched filtering SNR
    print("SNR of source", np.sqrt(SNR2))

    # Compute things in the wavelet domain

    signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
    psd = FrequencySeries(data=PSD, freq=freq)

    h_wavelet = from_freq_to_wavelet(signal_f_series, Nf=512)
    psd_wavelet = evolutionary_psd_from_stationary_psd(
        psd=psd.data,
        psd_f=psd.freq,
        f_grid=h_wavelet.freq,
        t_grid=h_wavelet.time,
        dt=delta_t,
    )

    SNR2_wavelet = np.nansum(((h_wavelet * h_wavelet) / psd_wavelet).data)
    print("SNR in wavelet domain is", SNR2_wavelet ** (1 / 2))

    # Gaps in the frequency domain.
    w_t = gap_routine(
        t_pad, start_window=4, end_window=6, lobe_length=1, delta_t=delta_t
    )

    freq_bins_pos_neg = np.fft.fftshift(np.fft.fftfreq(len(w_t), delta_t))
    freq_bins_pos_neg[N // 2] = freq_bins_pos_neg[N // 2 + 1]

    if TDI == "TDI1" or TDI == "TDI2":
        PSD_pos_neg = noise_PSD_AE(freq_bins_pos_neg, TDI=TDI)
    else:
        PSD_pos_neg = CornishPowerSpectralDensity(freq_bins_pos_neg)

    delta_f = freq[2] - freq[1]

    # - use positive and negative frequencies.
    w_fft = np.fft.fftshift(
        np.fft.fft(w_t)
    )  # Compute fft of windowing function (neg_pos_freq)
    w_star_fft = np.conjugate(w_fft)  # Compute conjugate

    Cov_Matrix = np.zeros(
        shape=(N // 2 + 1, N // 2 + 1), dtype=complex
    )  # Matrix will be filled full of complex numbers.
    # here we only have positive frequencies

    # Build analytical covariance matrix
    print("Building the analytical covariance matrix")
    Cov_Matrix_Gated = get_Cov(
        Cov_Matrix, w_fft, w_star_fft, delta_f, PSD_pos_neg
    )
    Cov_Matrix_Gated_Inv = np.linalg.inv(Cov_Matrix_Gated)

    # Regularise covariance matrix (stop it being singular)
    zero_points_window = np.argwhere(w_t == 0)[1][0]
    tol = w_t[zero_points_window - 1]  # Last moment before w_t is nonzero
    Cov_Matrix_Gated_Inv_Regularised = regularise_matrix(
        Cov_Matrix_Gated, w_t, tol=tol
    )

    Cov_Matrix_Stat = np.linalg.inv(np.diag(N * PSD / (2 * delta_t)))

    # =================== Compute SNRs ====================================
    h_w_t = h_t_pad * w_t
    h_w_fft = np.fft.rfft(h_w_t)

    SNR2_gaps = np.real((2 * h_w_fft.conj() @ Cov_Matrix_Gated_Inv @ h_w_fft))
    SNR2_gaps_regularised = np.real(
        (2 * h_w_fft.conj() @ Cov_Matrix_Gated_Inv_Regularised @ h_w_fft)
    )
    SNR2_no_gaps = np.real((2 * h_true_f.conj() @ Cov_Matrix_Stat @ h_true_f))

    print(
        "SNR when there are gaps in the frequency domain:",
        SNR2_gaps ** (1 / 2),
    )
    print(
        "SNR when there are gaps in the frequency domain using regularised matrix:",
        SNR2_gaps_regularised ** (1 / 2),
    )
    print(
        "SNR when there are no gaps in the frequency domain:",
        SNR2_no_gaps ** (1 / 2),
    )

    if TDI == "TDI1":
        np.save(
            f"{DATA_DIR}/Cov_Matrix_analytical_gap_TDI1.npy", Cov_Matrix_Gated
        )
        np.save(
            f"{DATA_DIR}/Cov_Matrix_analytical_inv_regularised_TDI1.npy",
            Cov_Matrix_Gated_Inv_Regularised,
        )
    elif TDI == "TDI2":
        np.save(
            f"{DATA_DIR}/Cov_Matrix_analytical_gap_TDI2.npy", Cov_Matrix_Gated
        )
        np.save(
            f"{DATA_DIR}/Cov_Matrix_analytical_inv_regularised_TDI2.npy",
            Cov_Matrix_Gated_Inv_Regularised,
        )
    else:
        np.save(
            f"{DATA_DIR}/Cov_Matrix_analytical_gap_Cornish.npy",
            Cov_Matrix_Gated,
        )
        np.save(
            f"{DATA_DIR}/Cov_Matrix_analytical_inv_regularised_Cornish.npy",
            Cov_Matrix_Gated_Inv_Regularised,
        )
