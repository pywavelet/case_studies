import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.transforms.to_wavelets import (from_freq_to_wavelet, from_time_to_wavelet)
from pywavelet.utils.lisa import zero_pad


def stitch_together_data_wavelet(w_t, t, h_pad_w, Nf, delta_t, start_window, end_window, windowing = False, alpha = 0, filter = False, **kwargs):
    start_index_gap = np.argwhere(np.isnan(w_t) == True)[0][0]
    end_index_gap = np.argwhere(np.isnan(w_t) == True)[-1][0]

    #================= PROCESS CHUNK 1 ===================
    kwgs_chunk_1 = dict(Nf = Nf)
    h_chunk_1 = h_pad_w[0:start_index_gap]
    if windowing == True:
        taper = tukey(len(h_chunk_1),alpha)
        h_chunk_1 = bandpass_data(taper * h_chunk_1, 5e-4, 1/delta_t, bandpassing_flag = filter, order = 2) 
    else:
        taper = tukey(len(h_chunk_1),0.0)
        h_chunk_1 = bandpass_data(taper * h_chunk_1, 5e-4, 1/delta_t, bandpassing_flag = filter, order = 2) 

    h_chunk_1_pad = zero_pad(h_chunk_1*taper)
    ND_1 = len(h_chunk_1_pad)
    Nf_1 = Nf
    kwgs_chunk_2 = dict(Nf = Nf_1)
    freq_bin_1 = np.fft.rfftfreq(ND_1, delta_t); freq_bin_1[0] = freq_bin_1[0]
    h_chunk_1_f = np.fft.rfft(h_chunk_1_pad)
    h_chunk_1_freq_series = FrequencySeries(h_chunk_1_f , freq = freq_bin_1)
    h_chunk_1_wavelet = from_freq_to_wavelet(h_chunk_1_freq_series, **kwgs_chunk_1)

    #================= PROCESS CHUNK 2 ===================
    h_chunk_2 = h_pad_w[end_index_gap+1:]
    if windowing == True:
        taper = tukey(len(h_chunk_2),alpha)
        h_chunk_2 = bandpass_data(taper * h_chunk_2, 5e-4, 1/delta_t, bandpassing_flag = filter, order = 2) 
    else:
        taper = tukey(len(h_chunk_2),0.0)
        h_chunk_2 = bandpass_data(taper * h_chunk_2, 5e-4, 1/delta_t, bandpassing_flag = filter, order = 2) 
    h_chunk_2_pad = zero_pad(h_chunk_2*taper)
    ND_2 = len(h_chunk_2_pad)
    Nf_2 = Nf
    kwgs_chunk_2 = dict(Nf = Nf_2)
    freq_bin_2 = np.fft.rfftfreq(ND_2, delta_t); freq_bin_2[0] = freq_bin_2[0]
    h_chunk_2_f = np.fft.rfft(h_chunk_2_pad)
    h_chunk_2_f_freq_series = FrequencySeries(h_chunk_2_f, freq = freq_bin_2)
    h_chunk_2_wavelet = from_freq_to_wavelet(h_chunk_2_f_freq_series, **kwgs_chunk_2)

    #===================== Now need to stitch together the data sets =========

    # Extract the time bins of wavelet data set 1
    wavelet_time_bins_chunk_1 = h_chunk_1_wavelet.time.data

    # Create mask. True for when wavelet times < start window. False otherwise
    mask_chunk_1 = wavelet_time_bins_chunk_1 < start_window * 60 * 60 

    # Force data within gap to be nans for chunk 1 
    h_chunk_1_w_nans = h_chunk_1_wavelet.data.copy()
    h_chunk_1_w_nans[:, ~mask_chunk_1] = np.nan

    # Chunk 2 has no gaps. The signal starts outside the gap and is just 
    # zero padded. This extra zero padding contributes nothing and we can
    # remove it to match with the original data set. 

    # Extract wavelet time bins full signal. This is still general, could get this
    # from the fitted window function. 
    w_t_TimeSeries = TimeSeries(w_t, t)
    w_t_wavelet = from_time_to_wavelet(w_t_TimeSeries, Nf)
    wavelet_times_full_signal = w_t_wavelet.time.data
    Nt = w_t_wavelet.data.shape[1] # Extract Nt of full data set 
    # Generate mask. True outside of gap, false inside gap
    mask = (wavelet_times_full_signal < start_window*60*60) | (wavelet_times_full_signal > end_window*60*60)
    # Compute number of indices where gap exists (count true number of nans in full data set )
    N_time_indices_for_gaps_full_signal = len(mask) - sum(mask)
    # Compute the number of indices where gap exists in chunk 1
    N_time_indicies_for_gaps_chunk_1 = len(mask_chunk_1) - sum(mask_chunk_1)
    # Compute number of extra gaps to append in total
    N_gap_to_append = N_time_indices_for_gaps_full_signal - N_time_indicies_for_gaps_chunk_1

    h_chunk_1_wavelet_matrix = h_chunk_1_wavelet.data
    # Create a matrix of NaNs with the same number of rows and N_gap_to_append columns
    nan_columns = np.full((h_chunk_1_wavelet_matrix.shape[0], N_gap_to_append), np.nan)
    h_chunk_1_wavelet_matrix_with_gaps = np.concatenate((h_chunk_1_w_nans, nan_columns), axis=1)

    # Concatenate chunk 1 and chunk 2. This will have new Nt dimensions larger than old Nt
    h_chunk_1_and_chunk_2 = np.concatenate((h_chunk_1_wavelet_matrix_with_gaps, h_chunk_2_wavelet.data), axis = 1)

    # Crop the remaining values of Nt. Due to zero padding, they're zero anyway. 
    h_approx_stitched_data = h_chunk_1_and_chunk_2[:, :Nt]
    # The matrix above has dimensions the same as the original data set :). Nf must be fixed. 
    # This is general and is the general structure from time -> freq -> wavelet for given gapped data set

    return h_approx_stitched_data, mask
def bandpass_data(rawstrain, f_min_bp, fs, bandpassing_flag = False, order = 4):
     """

    Bandpass the raw strain between [f_min, f_max] Hz.

    Arguments
    ---------

    rawstrain: numpy array
    The raw strain data.
    f_min_bp: float
    The lower frequency of the bandpass filter.
    f_max_bp: float
    The upper frequency of the bandpass filter.
    srate_dt: float
    The sampling rate of the data.
    bandpassing_flag: bool
    Whether to apply bandpassing or not.

    Returns
    -------

    strain: numpy array
    The bandpassed strain data.

    """
     if(bandpassing_flag):
     # Bandpassing section.
         # Create a fourth order Butterworth bandpass filter between [f_min, f_max] and apply it with the function filtfilt.
        #  bb, ab = butter(order, [f_min_bp/(0.5*srate_dt), f_max_bp/(0.5*srate_dt)], btype='band')

         f_nyq = 0.5 * fs
         bb, ab = butter(order, f_min_bp/(f_nyq), btype = "highpass")
         strain = filtfilt(bb, ab, rawstrain)
     else:
         strain = rawstrain
     return strain
