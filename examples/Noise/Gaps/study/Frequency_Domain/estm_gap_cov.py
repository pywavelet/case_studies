import os

import numpy as np
from scipy.signal.windows import tukey
from tqdm import tqdm

from gap_study_utils.gap_funcs import gap_routine, regularise_matrix
from gap_study_utils.noise_curves import noise_PSD_AE
from gap_study_utils.signal_utils import inner_prod, waveform, zero_pad

np.random.seed(1234)

DATA_DIR = os.path.join(os.path.dirname(__file__), "Data")


def main(
    a_true=1e-21,
    f_true=3e-3,
    fdot_true=1e-8,
    TDI="TDI1",  # TDI1 red noise, TDI2 white noise at low f
    tmax=10 * 60 * 60,  # Final time
):

    # error if TDI not in ['TDI1', 'TDI2']
    if TDI not in ["TDI1", "TDI2"]:
        raise ValueError("TDI must be either 'TDI1' or 'TDI2', not {TDI}")

    fs = 2 * f_true  # Sampling rate
    delta_t = np.floor(0.4 / fs)  # Sampling interval

    t = np.arange(
        0, tmax, delta_t
    )  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

    # Round length of time series to a power of two.
    N = int(2 ** (np.ceil(np.log2(len(t)))))

    # Gen waveform, taper and then pad
    h_t = waveform(a_true, f_true, fdot_true, t)
    taper_signal = tukey(len(h_t), alpha=0.2)
    h_t_pad = zero_pad(h_t * taper_signal)

    # Compute new time array corresponding to padded signal
    t_pad = np.arange(0, len(h_t_pad) * delta_t, delta_t)

    # Fourier domain, gen frequencies, PSD and signal in freq
    freq = np.fft.rfftfreq(N, delta_t)
    freq[0] = freq[1]  # set f_0 = f_1 for PSD
    PSD = noise_PSD_AE(freq, TDI=TDI)
    h_true_f = np.fft.rfft(h_t_pad)

    # Compute SNR
    SNR2 = inner_prod(h_true_f, h_true_f, PSD, delta_t, N)
    print("SNR of source", np.sqrt(SNR2))

    # Gaps in the frequency domain.

    # Generate gaps and apply them to signal
    w_t = gap_routine(
        t_pad, start_window=4, end_window=6, lobe_length=1, delta_t=delta_t
    )
    h_w_pad = w_t * h_t_pad
    h_w_fft = np.fft.rfft(h_w_pad)
    variance_noise_f = (
        N * PSD / (4 * delta_t)
    )  # Compute variance in frequency domain (pos freq)

    # Generate covariance of noise for stat process
    cov_matrix_stat = np.linalg.inv(np.diag(2 * variance_noise_f))

    # ====================== ESTIMATE THE NOISE COVARIANCE MATRIX ==============================
    print("Estimating the gated covariance matrix")
    noise_f_gap_vec = []
    for i in tqdm(range(0, 5000)):
        np.random.seed(i)
        noise_f_iter = np.random.normal(
            0, np.sqrt(variance_noise_f)
        ) + 1j * np.random.normal(0, np.sqrt(variance_noise_f))
        noise_f_iter[0] = np.sqrt(2) * noise_f_iter[0].real
        noise_f_iter[-1] = np.sqrt(2) * noise_f_iter[-1].real

        noise_t_iter = np.fft.irfft(
            noise_f_iter
        )  # Compute stationary noise in TD
        noise_t_gap_iter = (
            w_t * noise_t_iter
        )  # Place gaps in the noise from the TD
        noise_f_gap_iter = np.fft.rfft(noise_t_gap_iter)  # Convert into FD
        noise_f_gap_vec.append(noise_f_gap_iter)

        # ==========================================================================================

    print("Now estimating covariance matrix")
    cov_matrix_freq_gap = np.cov(noise_f_gap_vec, rowvar=False)

    # Regularise covariance matrix (stop it being singular)
    zero_points_window = np.argwhere(w_t == 0)[1][0]
    tol = w_t[zero_points_window - 1]  # Last moment before w_t is nonzero
    cov_matrix_freq_gap_regularised_inv = regularise_matrix(
        cov_matrix_freq_gap, w_t, tol=tol
    )
    print("Finished estimating the covariance matrix")

    # cov matrix freq gap
    cov_matrix_freq_gap_inv = np.linalg.inv(
        cov_matrix_freq_gap
    )  # Compute inverse of estimated matrix

    # Compute various SNRs
    SNR2_gaps = np.real(
        (2 * h_w_fft.conj() @ cov_matrix_freq_gap_inv @ h_w_fft)
    )
    SNR2_gaps_regularised = np.real(
        (2 * h_w_fft.conj() @ cov_matrix_freq_gap_regularised_inv @ h_w_fft)
    )
    SNR2_no_gaps = np.real((2 * h_true_f.conj() @ cov_matrix_stat @ h_true_f))

    print("SNR when there are gaps in the frequency domain", SNR2_gaps**0.5)
    print(
        "SNR when there are gaps in the frequency domain is",
        SNR2_gaps_regularised**0.5,
        "using regularised matrix",
    )
    print(
        "SNR when there are no gaps in the frequency domain", SNR2_no_gaps**0.5
    )

    # Save the data
    np.save(f"{DATA_DIR}/Cov_Matrix_estm_gap_{TDI}.npy", cov_matrix_freq_gap)
    np.save(
        f"{DATA_DIR}/Cov_Matrix_estm_inv_regularised_{TDI}.npy",
        cov_matrix_freq_gap_regularised_inv,
    )


if __name__ == "__main__":
    main()
