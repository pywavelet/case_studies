import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.transforms.to_wavelets import (from_freq_to_wavelet, from_time_to_wavelet)
from pywavelet.utils.lisa import zero_pad


import matplotlib.pyplot as plt

def stitch_together_data_wavelet(w_t, t, h_pad_w, Nf, delta_t, start_window, 
                                end_window, windowing = False, alpha = 0, 
                                filter = False):
    """
    Stitch together data segments in the wavelet domain, filling gaps with NaNs.

    This function processes a data stream that is divided into chunks due to gaps. 
    It converts each chunk into the wavelet domain, applies windowing and filtering if specified, 
    and then stitches the chunks together, filling the gaps with NaNs.

    Parameters
    ----------
    w_t : array_like
        Original time series data with gaps filled with NaNs.
    t : array_like
        Time array corresponding to the original time series data.
    h_pad_w : array_like
        Padded data stream in the wavelet domain.
    Nf : int
        Number of frequency bins for the wavelet transform.
    delta_t : float
        Time step between samples in the original time series.
    start_window : float
        Start time of the gap window in hours.
    end_window : float
        End time of the gap window in hours.
    windowing : bool, optional
        If True, apply a Tukey window to the data chunks before processing (default is False).
    alpha : float, optional
        Shape parameter for the Tukey window (default is 0).
    filter : bool, optional
        If True, apply a high-pass Butterworth filter to the data chunks (default is False).

    Returns
    -------
    h_approx_stitched_data : np.ndarray
        Stitched data in the wavelet domain, with gaps filled with NaNs.
    mask : np.ndarray
        Boolean mask indicating the positions of the gaps in the time domain.

    Notes
    -----
    This function assumes that the wavelet coefficients of the data are Gaussian.

    Steps:
    1. Determine the indices where the gaps start and end in the original time series.
    2. Process each chunk of the data:
    - Apply a Tukey window if specified.
    - Apply a high-pass Butterworth filter if specified.
    - Convert the data chunk to the wavelet domain.
    3. Stitch the processed chunks together:
    - Fill the gap in the first chunk with NaNs.
    - Append the necessary number of NaNs to match the original time series length.
    4. Return the stitched data and the mask indicating the gap positions.
    """
    # calculate the index (in normal time) where the gap starts
    # and ends 

    start_index_gap = np.argwhere(np.isnan(w_t) == True)[0][0]
    end_index_gap = np.argwhere(np.isnan(w_t) == True)[-1][0]
    N_end = len(t) #int(t[-1]/delta_t)
    f_min = 9e-4

    #================= PROCESS CHUNK 1 ===================
    kwgs_wavelet = dict(Nf = Nf)
    # Chunk the data stream into data stream 1
    h_chunk_1 = h_pad_w[0:start_index_gap]

    # If we window, then we will apply a high pass filer to the tapered signal.
    # This is essential if we want to work with noise from a very large red 
    # noise process
    taper_1 = tukey(len(h_chunk_1),alpha)
    if filter == True:
        h_chunk_1 = bandpass_data(h_chunk_1, f_min, 1/delta_t, bandpassing_flag = filter, order = 4) 
    if windowing == True:
        h_chunk_1 *= taper_1
    h_chunk_1_pad = zero_pad(h_chunk_1) # Zero pad the data stream
    ND_1 = len(h_chunk_1_pad) # Length of data stream for chunk 1

    # Convert chunk 1 into wavelet data stream 
    freq_bin_1 = np.fft.rfftfreq(ND_1, delta_t); freq_bin_1[0] = freq_bin_1[0]
    h_chunk_1_f = np.fft.rfft(h_chunk_1_pad)
    h_chunk_1_freq_series = FrequencySeries(h_chunk_1_f , freq = freq_bin_1)
    h_chunk_1_wavelet = from_freq_to_wavelet(h_chunk_1_freq_series, **kwgs_wavelet)

    
    #================= PROCESS CHUNK 2, end of data ===================
    # This is exactly the same as the lines above
    # The goal here is to turn data stream 2 into wavelet domain
    h_chunk_2 = h_pad_w[end_window:N_end]
    h_chunk_2_padded = h_pad_w[end_index_gap:]
    N_final_chunk = len(zero_pad(h_chunk_2_padded))

    taper_2 = tukey(len(h_chunk_2),alpha)
    if filter == True:
        h_chunk_2 = bandpass_data(h_chunk_2, f_min, 1/delta_t, bandpassing_flag = filter, order = 4) 
    if windowing == True:
        h_chunk_2 *= taper_2

    padding_length = N_final_chunk - len(h_chunk_2)
    h_chunk_2_pad = np.pad(h_chunk_2, (0, padding_length), 'constant', constant_values=0)

    ND_2 = len(h_chunk_2_pad)
    freq_bin_2 = np.fft.rfftfreq(ND_2, delta_t); freq_bin_2[0] = freq_bin_2[0]
    h_chunk_2_f = np.fft.rfft(h_chunk_2_pad)
    h_chunk_2_f_freq_series = FrequencySeries(h_chunk_2_f, freq = freq_bin_2)
    h_chunk_2_wavelet = from_freq_to_wavelet(h_chunk_2_f_freq_series, **kwgs_wavelet)

    #===================== Build window to pass to templates ==========

    # taper_1_padded = np.pad(taper_1[0:-1], (0, end_index_gap - start_index_gap), 'constant', constant_values=0)
    taper_1_padded = np.pad(taper_1, (0, end_index_gap - start_index_gap + 1), 'constant', constant_values=0)
    template_window = np.concatenate((taper_1_padded,taper_2))
    template_window_pad = zero_pad(template_window)
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
    t_pad = np.arange(0,len(w_t)*delta_t, delta_t)
    w_t_TimeSeries = TimeSeries(w_t, t_pad)
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

    return h_approx_stitched_data, mask, template_window_pad
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
         f_nyq = 0.5 * fs
         bb, ab = butter(order, f_min_bp/(f_nyq), btype = "highpass")
         strain = filtfilt(bb, ab, rawstrain)
     else:
         strain = rawstrain
     return strain
