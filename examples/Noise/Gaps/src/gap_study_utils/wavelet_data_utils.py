import numpy as np


from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.transforms import (from_freq_to_wavelet, from_time_to_wavelet)
from pywavelet.transforms.forward.wavelet_bins import compute_bins

from .gap_window import GapWindow
from .signal_utils import waveform_generator
from typing import List, Optional



def chunk_timeseries(ht:TimeSeries, gaps:List[GapWindow], windowing_alpha:float=0, filter:bool=False, fmin:float=7e-4)->List[TimeSeries]:
    """
    Split a TimeSeries object into  chunks based on the gaps


    If N gaps, procduces N+1 chunks

    """

    N_gaps = len(gaps)

    # Sort gaps by their start index to ensure ordered splitting
    gaps = sorted(gaps, key=lambda g: g.start_idx)

    # ensure that no gaps overlap
    for i in range(len(gaps) - 1):
        assert gaps[i].end_idx < gaps[i + 1].start_idx


    chunks = []

    # Starting index of the first chunk
    start_idx = 0

    for gap in gaps:
        # Create chunk before the current gap
        chunks.append(TimeSeries(ht.data[start_idx:gap.start_idx], ht.time[start_idx:gap.start_idx]))

        # Update start index for the next chunk
        start_idx = gap.end_idx + 1

    # Add the remaining data as the last chunk
    if start_idx < len(ht.data):
        chunks.append(TimeSeries(ht.data[start_idx:], ht.time[start_idx:]))

    assert len(chunks) == N_gaps + 1

    # Apply zero-padding, windowing, and filtering to each chunk
    for i in range(len(chunks)):
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
        gaps: List[GapWindow],
        ht: TimeSeries,
        Nf: int,
        alpha: float = 0.0,
        fmin: float = 7e-4,
        filter: bool = False,
) -> Wavelet:
    # Split into chunks based on multiple gaps
    chunked_timeseries = chunk_timeseries(ht, gaps, windowing_alpha=alpha, filter=filter, fmin=fmin)

    # Convert each chunk to wavelet and apply NaNs for gaps
    chunked_wavelets = []
    for chunk in chunked_timeseries:
        w = from_freq_to_wavelet(chunk.to_frequencyseries(), Nf)
        for gap in gaps:
            w = gap.apply_nan_gap_to_wavelet(w)
        chunked_wavelets.append(w)

    # Setting up the final wavelet data array
    Nt = ht.ND // Nf
    time_bins, freq_bins = compute_bins(Nf, Nt, ht.duration)
    stiched_data = np.full((Nf, Nt), np.nan)

    # Fill in data from each wavelet chunk, handling the time alignment
    for i, w in enumerate(chunked_wavelets):
        # Get indices for matching time_bins with wavelet time
        stich_tmask = np.zeros(Nt, dtype=bool)
        stich_tmask[np.argmin(np.abs(time_bins[:, None] - w.time), axis=0)] = True

        # Get mask for valid time values in the wavelet
        w_tmask = np.zeros(w.Nt, dtype=bool)
        w_tmask[np.argmin(np.abs(w.time[:, None] - time_bins), axis=0)] = True

        # Apply chunk data to final wavelet data array
        stiched_data[:, stich_tmask] = chunked_wavelets[i].data[:, w_tmask]

    # Truncate data up to tmax if specified
    tmask = time_bins <= max(gap.tmax for gap in gaps)
    return Wavelet(stiched_data[:, tmask], time_bins[tmask], freq_bins)

