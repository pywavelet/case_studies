import numpy as np
from scipy.signal.windows import tukey
from scipy.signal import butter, filtfilt
from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.transforms import (from_freq_to_wavelet, from_time_to_wavelet)
from pywavelet.transforms.forward.wavelet_bins import compute_bins

from .gap_funcs import GapWindow
from .signal_utils import zero_pad, waveform_generator
from typing import List, Optional

import matplotlib.pyplot as plt


def chunk_timeseries(ht:TimeSeries, gap:GapWindow, windowing_alpha:float=0, filter:bool=False)->List[TimeSeries]:
    """
    Split a TimeSeries object into two chunks based on the gap window.
    """
    chunks = [
        TimeSeries(ht.data[0:gap.start_idx],ht.time[0:gap.start_idx]),
        TimeSeries(ht.data[gap.end_idx + 1:], ht.time[gap.end_idx + 1:])
    ]

    for i in range(2):
        d = chunks[i].data

        if filter:
            taper = tukey(len(d), alpha=windowing_alpha)
            d = bandpass_data(d*taper, f_min_bp=7e-4, fs=ht.fs,  order=4)

        d = zero_pad(d)
        t = np.arange(0, len(d) * ht.dt, ht.dt) + chunks[i].time[0]
        chunks[i] = TimeSeries(d, t)

    return chunks


def gap_hwavelet_generator(a:float, ln_f:float, ln_fdot:float, time:np.ndarray, gap:Optional[GapWindow], tmax:float, Nf:int, alpha=0.0, filter=False)->Wavelet:
    f, fdot = np.exp(ln_f), np.exp(ln_fdot)
    ht = waveform_generator(a, f, fdot, time, tmax, alpha=alpha)
    if gap is not None:
        hwavelet =  generate_wavelet_with_gap(gap, ht, Nf, alpha=alpha, filter=filter)
    else:
        hwavelet = from_freq_to_wavelet(ht.to_frequencyseries(), Nf)
    return hwavelet




def generate_wavelet_with_gap(
        gap: GapWindow,
        ht:TimeSeries,
        Nf: int,
        alpha:float=0.0,
        filter=False,
):
    chunked_timeseries = chunk_timeseries(ht, gap, windowing_alpha=alpha, filter=filter)
    chunked_wavelets = [from_freq_to_wavelet(chunk.to_frequencyseries(), Nf) for chunk in chunked_timeseries]
    # ensure any gaps are filled with nans and truncate wavelets to the correct time
    # (i.e. get rid of padding added for FFT)
    chunked_wavelets = [gap.apply_nan_gap_to_wavelet(w) for w in chunked_wavelets]

    # setting up final wavelet
    Nt = ht.ND // Nf
    time_bins, freq_bins = compute_bins(Nf, Nt, ht.duration)
    stiched_data = np.full((Nf, Nt), np.nan)


    for i, w in enumerate(chunked_wavelets):
        # get idx of the time_bins that correspond to the t
        stich_tmask = np.zeros(Nt, dtype=bool)
        stich_tmask[np.argmin(np.abs(time_bins[:, None] - w.time), axis=0)] = True

        # get mask to ensure we only use t values inside time_bins
        w_tmask = np.zeros(w.Nt, dtype=bool)
        w_tmask[np.argmin(np.abs(w.time[:, None] - time_bins), axis=0)] = True

        # check no values already in that region
        # assert np.all(np.isnan(stiched_data[:, stich_tmask]))
        # fill in the values from the chunked wavelet
        stiched_data[:, stich_tmask] = chunked_wavelets[i].data[:, w_tmask]


    # onnly keep data up to tmax
    tmask = time_bins <= gap.tmax
    return Wavelet(stiched_data[:, tmask], time_bins[tmask], freq_bins)


def bandpass_data(rawstrain:np.ndarray, f_min_bp:float, fs:float, order:int=4):
    """

   Bandpass the raw strain between [f_min, f_max] Hz.


    Create a Nth order Butterworth bandpass filter between [f_min, f_max] and apply it with the function filtfilt.
        #  bb, ab = butter(order, [f_min_bp/(0.5*srate_dt), f_max_bp/(0.5*srate_dt)], btype='band')


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

   Returns
   -------

   strain: numpy array
   The bandpassed strain data.

   """
    f_nyq = 0.5 * fs
    bb, ab = butter(order, f_min_bp / (f_nyq), btype="highpass")
    strain = filtfilt(bb, ab, rawstrain)
    return strain
