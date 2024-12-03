from enum import Enum
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from pywavelet.transforms.forward.wavelet_bins import compute_bins
from pywavelet.transforms.types import TimeSeries, Wavelet
from pywavelet.transforms.types.plotting import _fmt_time_axis


class GapType(Enum):
    STITCH = 1
    RECTANGULAR_WINDOW = 2

    def __repr__(self):
        # name without the class name
        return self.name.split(".")[-1]

    def __str__(self):
        return self.__repr__()

    @staticmethod
    def all_types() -> List["GapType"]:
        return [*GapType]


class GapWindow:
    def __init__(
        self,
        time: np.array,
        gap_ranges: List[Tuple[float, float]],
        tmax: float,
        type: GapType = GapType.STITCH,
    ):
        """
        Args:
            time (np.array):
                Time array representing the time points of the data.
            gap_ranges (List[Tuple[float, float]]):
                List of tuples where each tuple represents the start and end time of a gap.
                Example: [(start1, end1), (start2, end2), ...]
            tmax (float):
                Maximum time value of the time array. This is used to define the upper limit of the time series.
            type (GapType):
                Type of gap window to apply. It can be either GapType.STITCH or GapType.RECTANGULAR_WINDOW.
        """
        self.__overlap_check(gap_ranges)
        self.gap_ranges = gap_ranges
        self.n_gaps = len(gap_ranges)
        self.nan_mask = self.__generate_nan_mask(time, gap_ranges)
        self.gap_bools = np.isnan(
            self.nan_mask
        )  # True for valid data, False for gaps
        self.time = time
        self.t0: float = time[0]
        self.tmax = (
            tmax  # Note-- t[-1] is not necessarily tmax -- might be padded.
        )
        self.start_idxs = [
            np.argmin(np.abs(time - start)) for start, _ in gap_ranges
        ]
        self.end_idxs = [
            np.argmin(np.abs(time - end)) - 1 for _, end in gap_ranges
        ]
        self.type = type

    def non_gap_idxs(self) -> List[Tuple[int, int]]:
        """
        Returns:
            List[List[int]]: List [[start, end]] of non-gap indices (N_gaps + 1)
        """
        idxs = []
        data_start = 0
        for gap_start, gap_end in self.gap_idxs():
            idxs.append((data_start, gap_start - 1))
            data_start = gap_end + 1
        idxs.append((data_start, len(self.time) - 1))
        return idxs

    def gap_idxs(self) -> List[Tuple[int, int]]:
        """
        Returns:
            List[List[int]]: List [[start, end]] of gap indices (N_gaps)
        """
        return list(zip(self.start_idxs, self.end_idxs))

    def __len__(self):
        return len(self.nan_mask)

    def gap_len(self):
        return np.sum(np.isnan(self.nan_mask))

    @property
    def num_nans(self):
        return np.sum(np.isnan(self.nan_mask))

    @property
    def fraction_nans(self):
        return self.num_nans / len(self.nan_mask)

    def __repr__(self):
        return f"GapWindow({self.type}, {self.num_nans:,}/{len(self.nan_mask):,} NaNs)"

    @staticmethod
    def __generate_nan_mask(
        t: np.ndarray, gap_ranges: List[Tuple[float, float]]
    ) -> np.ndarray:
        """Returns [1,1,1,0,0,0,1,1,1] , where nan is in the gap"""
        nan_mask = np.ones_like(t, dtype=float)
        for start_window, end_window in gap_ranges:
            mask = (t > start_window) & (t < end_window)
            nan_mask[mask] = np.nan
        return nan_mask

    def apply_window(self, timeseries) -> TimeSeries:
        return TimeSeries(timeseries.data * self.nan_mask, self.time)

    def apply_nan_gap_to_wavelet(self, w: Wavelet) -> Wavelet:
        data, t = w.data.copy(), w.time
        nan_mask = self.get_nan_mask_for_timeseries(t)
        time_mask = self.inside_timeseries(t)
        data[:, nan_mask] = np.nan
        return Wavelet(data[:, time_mask], t[time_mask], w.freq)

    def get_nan_mask_for_timeseries(self, t) -> Union[bool, np.ndarray]:
        return ~self.valid_t(t)

    def inside_gap(self, t: float) -> bool:
        mask = np.zeros_like(t, dtype=bool)
        for gap_start, gap_end in self.gap_ranges:
            mask |= (gap_start <= t) & (t <= gap_end)
        return mask

    def inside_timeseries(self, t: np.ndarray) -> bool:
        return (self.t0 <= t) & (t <= self.tmax)

    def valid_t(self, t: Union[float, np.ndarray]) -> bool:
        return self.inside_timeseries(t) & ~self.inside_gap(t)

    def plot(self, ax: plt.Axes = None, **kwgs) -> Union[None, plt.Axes]:
        if ax is None:
            fig, ax = plt.subplots()

        kwgs["alpha"] = kwgs.get("alpha", 0.2)
        kwgs["edgecolor"] = kwgs.get("edgecolor", "gray")
        kwgs["hatch"] = kwgs.get("hatch", "/")
        kwgs["zorder"] = kwgs.get("zorder", 10)
        kwgs["fill"] = kwgs.get("fill", False)

        for gap_start, gap_end in self.gap_ranges:
            ax.axvspan(gap_start, gap_end, **kwgs)
        _fmt_time_axis(self.time, ax, self.t0, self.tmax)
        return ax

    @staticmethod
    def __overlap_check(gap_ranges: List[Tuple[float, float]]):
        for i in range(len(gap_ranges) - 1):
            if gap_ranges[i][1] > gap_ranges[i + 1][0]:
                raise ValueError(
                    "Gap ranges must not overlap. Found overlap between "
                    f"{gap_ranges[i]} and {gap_ranges[i + 1]}"
                )

    def _chunk_timeseries(
        self,
        ht: TimeSeries,
        alpha: float = 0,
        fmin: float = 0,
    ) -> List[TimeSeries]:
        """
        Split a TimeSeries object into chunks based on the gaps.
        Returns:
            Ngaps+1 time series chunks
        """
        chunks = []
        for i, (t0_idx, tend_idx) in enumerate(self.non_gap_idxs()):
            ts = ht[t0_idx:tend_idx].zero_pad_to_power_of_2(alpha)
            if fmin != 0:
                ts.highpass_filter(fmin, alpha)
            chunks.append(ts)
        return chunks

    def __gap_timeseries_chunk_transform_wdm_n_stitch(
        self,
        ht: TimeSeries,
        Nf: int,
        alpha: float = 0.0,
        fmin: float = 0,
    ) -> Wavelet:
        # Split into chunks and apply wavelet transform each chunk
        chunked_wavelets = [
            self.apply_nan_gap_to_wavelet(c.to_wavelet(Nf=Nf))
            for c in self._chunk_timeseries(ht, alpha, fmin)
        ]

        # Setting up the final wavelet data array
        Nt = ht.ND // Nf
        time_bins, freq_bins = compute_bins(Nf, Nt, ht.duration)
        stitched_data = np.full((Nf, Nt), np.nan)

        # Fill in data from each wavelet chunk, handling the time alignment
        for i, w in enumerate(chunked_wavelets):
            # Get indices for matching time_bins with wavelet time
            stich_tmask = np.zeros(Nt, dtype=bool)
            stich_tmask[
                np.argmin(np.abs(time_bins[:, None] - w.time), axis=0)
            ] = True

            # Get mask for valid time values in the wavelet
            w_tmask = np.zeros(w.Nt, dtype=bool)
            w_tmask[np.argmin(np.abs(w.time[:, None] - time_bins), axis=0)] = (
                True
            )

            # Apply chunk data to final wavelet data array
            stitched_data[:, stich_tmask] = chunked_wavelets[i].data[
                :, w_tmask
            ]

        # Truncate data up to tmax if specified
        tmask = time_bins <= self.tmax
        return Wavelet(stitched_data[:, tmask], time_bins[tmask], freq_bins)

    def __gap_timeseries_with_0s_n_transform(
        self, ht: TimeSeries, Nf: int, alpha: float = 0.0, fmin: float = 0
    ) -> Wavelet:
        if fmin != 0:
            ht.highpass_filter(fmin, alpha)
        # convert gapped timepoints to 0s
        ht.data[self.gap_bools] = 0
        # Apply wavelet transform
        w = ht.to_wavelet(Nf=Nf)
        # ensure regions in the gap are set to nan
        w = self.apply_nan_gap_to_wavelet(w)
        return w

    def gap_n_transform_timeseries(
        self, ht: TimeSeries, Nf: int, alpha: float = 0.0, fmin: float = 0
    ) -> Wavelet:
        if self.type == GapType.STITCH:
            return self.__gap_timeseries_chunk_transform_wdm_n_stitch(
                ht, Nf, alpha, fmin
            )
        elif self.type == GapType.RECTANGULAR_WINDOW:
            return self.__gap_timeseries_with_0s_n_transform(
                ht, Nf, alpha, fmin
            )
        else:
            raise ValueError(f"GapType {self.type} not recognized")
