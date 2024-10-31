import numpy as np


from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.transforms import (from_freq_to_wavelet, from_time_to_wavelet)
from pywavelet.transforms.forward.wavelet_bins import compute_bins

from .gap_funcs import GapWindow
from .signal_utils import waveform_generator
from typing import List, Optional



def chunk_timeseries(ht:TimeSeries, gap:GapWindow, windowing_alpha:float=0, filter:bool=False, fmin:float=7e-4)->List[TimeSeries]:
    """
    Split a TimeSeries object into two chunks based on the gap window.
    """
    chunks = [
        TimeSeries(ht.data[0:gap.start_idx],ht.time[0:gap.start_idx]),
        TimeSeries(ht.data[gap.end_idx + 1:], ht.time[gap.end_idx + 1:])
    ]

    for i in range(2):
        timeseries = chunks[i]
        timeseries = timeseries.zero_pad_to_power_of_2(tukey_window_alpha=windowing_alpha)
        if filter:
            timeseries.highpass_filter(
                fmin=fmin,
                tukey_window_alpha=windowing_alpha,
            )
        d = timeseries.data
        t = np.arange(0, len(d) * ht.dt, ht.dt) + chunks[i].time[0]
        chunks[i] = TimeSeries(d, t)

    return chunks


def gap_hwavelet_generator(
        a:float,
        ln_f:float,
        ln_fdot:float,
        time:np.ndarray,
        gap:Optional[GapWindow],
        tmax:float,
        Nf:int,
        alpha=0.0,
        filter=False,
        fmin:float=0.0
)->Wavelet:
    f, fdot = np.exp(ln_f), np.exp(ln_fdot)
    ht = waveform_generator(a, f, fdot, time, tmax, alpha=alpha)
    if gap is not None:
        hwavelet =  generate_wavelet_with_gap(gap, ht, Nf, alpha=alpha, filter=filter, fmin=fmin)
    else:
        if filter:
            ht = ht.highpass_filter(fmin=fmin, tukey_window_alpha=alpha)
        hwavelet = ht.to_wavelet(Nf=Nf)
    return hwavelet




def generate_wavelet_with_gap(
        gap: GapWindow,
        ht:TimeSeries,
        Nf: int,
        alpha:float=0.0,
        fmin:float=7e-4,
        filter=False,
):
    chunked_timeseries = chunk_timeseries(ht, gap, windowing_alpha=alpha, filter=filter, fmin=fmin)
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

        # fill in the values from the chunked wavelet
        stiched_data[:, stich_tmask] = chunked_wavelets[i].data[:, w_tmask]

    # only keep data up to tmax
    tmask = time_bins <= gap.tmax
    return Wavelet(stiched_data[:, tmask], time_bins[tmask], freq_bins)

#
# def bandpass_data(rawstrain:np.ndarray, f_min_bp:float, fs:float, order:int=4):
#     """
##    DONE IN pywavelet.transforms.types.TimeSeries.highpass_filter