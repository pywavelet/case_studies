from numba import njit, prange
from pywavelet.transforms.types import TimeSeries, Wavelet
from pywavelet.transforms.types.plotting import _fmt_time_axis
import numpy as np
import matplotlib.pyplot as plt

from typing import List


class GapWindow:
    def __init__(self, t: np.ndarray, gap_range: List[float], tmax: float):
        self.gap_start = gap_range[0]
        self.gap_end = gap_range[1]
        self.nan_mask = self.__generate_nan_mask(t, self.gap_start, self.gap_end)
        self.t = t
        self.tmax = tmax  # Note-- t[-1] is not necessarily tmax -- might be padded.
        # first idx od nan
        self.start_idx = np.argmax(self.nan_mask)
        # last idx of nanx
        self.end_idx = len(self.nan_mask) - np.argmax(self.nan_mask[::-1]) - 1

    def __len__(self):
        return len(self.nan_mask)

    def gap_len(self):
        # number of nan values
        return np.sum(np.isnan(self.nan_mask))

    @property
    def num_nans(self):
        return np.sum(np.isnan(self.nan_mask))

    @property
    def fraction_nans(self):
        return self.num_nans / len(self.nan_mask)

    def __repr__(self):
        return f"GapWindow({self.num_nans:,}/{len(self.nan_mask):,} NaNs)"

    @staticmethod
    def __generate_nan_mask(t: np.ndarray, start_window: float, end_window: float):
        """
        Insert NaN values in a given time window to simulate gaps in the data.

        Parameters:
        - t: Time array.
        - start_window: Start of the gap window in hours.
        - end_window: End of the gap window in hours.
        - delta_t: Time step.

        Returns:
        - nan_window:
            List with 1s for valid data and NaNs for missing data in the gap.
            [1, 1, 1, ..., NaN, NaN, ..., 1, 1, 1] for
            [t0...t_start_window, ..t_end_window, ...t_end]

        """
        return [np.nan if start_window < time < end_window else 1 for time in t]

    def apply_window(self, timeseries) -> TimeSeries:
        """
        Apply the gap window to a given time series.

        Parameters:
        - timeseries: Time series to apply the gap window.

        Returns:
        - TimeSeries:
            Time series with NaN values in the gap window.

        """
        return TimeSeries(timeseries.data * self.nan_mask, self.t)

    def apply_nan_gap_to_wavelet(self, w: Wavelet):
        """
        Apply the gap window to a given wavelet.

        Parameters:
        - w: Wavelet to apply the gap window.

        Returns:
        - Wavelet:
            Wavelet with NaN values in the gap window.

        """
        data, t = w.data.copy(), w.time
        nan_mask = self.get_nan_mask_for_timeseries(t)
        time_mask = self.inside_timeseries(t)
        data[:, nan_mask] = np.nan
        return Wavelet(data[:, time_mask], t[time_mask], w.freq)

    def get_nan_mask_for_timeseries(self, t):
        """Gets a mask for the given time series -- True for NaNs, False for valid data."""
        return ~self.valid_t(t)

    def inside_gap(self, t: float):
        # if t inside [gap_start, gap_end], True
        # else False
        return (self.gap_start <= t) & (t <= self.gap_end)

    def inside_timeseries(self, t: np.ndarray):
        return (self.t[0] <= t) & (t <= self.tmax)

    def valid_t(self, t: float):
        # if t inside timeseries and outside gap, True
        return self.inside_timeseries(t) & ~ self.inside_gap(t)

    def plot(self, ax: plt.Axes = None):
        if ax is None:
            fig, ax = plt.subplots()
        # axvspans across gap windows
        ax.axvspan(self.gap_start, self.gap_end, color="gray", alpha=0.2)
        _fmt_time_axis(self.t, ax)
        ax.legend()
        return ax
