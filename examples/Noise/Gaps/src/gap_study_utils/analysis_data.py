from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from pywavelet.transforms.forward.wavelet_bins import compute_bins
from pywavelet.transforms.types import FrequencySeries, TimeSeries, Wavelet
from pywavelet.utils import (
    compute_likelihood,
    compute_snr,
    evolutionary_psd_from_stationary_psd,
)

from .constants import DT, GAP_RANGES, NF, TMAX, TRUES
from .gap_window import GapType, GapWindow
from .noise_curves import (
    CornishPowerSpectralDensity,
    generate_stationary_noise,
)
from .signal_utils import waveform


class AnalysisData:
    """
    Encapsulates data and methods required for a gap study on time series and
    frequency analysis with wavelets, handling configurations for data sampling,
    PSDs, and gap-windowing, with optional plotting.
    """

    @classmethod
    def DEFAULT(cls):
        data_kwargs = {
            "dt": DT,
            "tmax": TMAX,
            "alpha": 0.0,
            "highpass_fmin": 0,
            "noise": False,
        }
        gap_kwargs = {
            "type": GapType.STITCH,
            "gap_ranges": GAP_RANGES,
        }
        waveform_generator = waveform
        waveform_parameters = TRUES
        return cls(
            data_kwargs, gap_kwargs, waveform_generator, waveform_parameters
        )

    def __init__(
        self,
        data_kwargs: Optional[Dict] = None,
        gap_kwargs: Optional[Dict] = None,
        waveform_generator: Optional[Callable[..., TimeSeries]] = None,
        waveform_parameters: Optional[List[float]] = None,
        plotfn: Optional[str] = None,
    ):
        self.data_kwargs = data_kwargs or {}
        self.gap_kwargs = gap_kwargs or {}

        breakpoint()
        self.waveform_generator = waveform_generator
        self.waveform_parameters = waveform_parameters
        self._initialize_data_params()
        self._initialize_grids()
        self._initialize_gap_window()

        if plotfn:
            self.plot_data(plotfn)

    def _initialize_data_params(self):
        """Initialize core data parameters from `data_kwargs` with default values."""
        self.dt = self.data_kwargs.get("dt", DT)
        self.tmax = self.data_kwargs.get("tmax", TMAX)
        self.alpha = self.data_kwargs.get("alpha", 0.0)
        self.highpass_fmin = self.data_kwargs.get("highpass_fmin", 0)
        self.noise = self.data_kwargs.get("noise", False)
        self.ND = int(self.tmax / self.dt)
        self.time = np.arange(0, self.tmax, self.dt)
        self.freq = np.fft.rfftfreq(self.ND, d=self.dt)

    def _initialize_grids(self):
        """Compute time and frequency grids for wavelet analysis."""
        self.Nf = self.data_kwargs.get("Nf", NF)
        self.Nt = self.ND // self.Nf
        self.t_grid, self.f_grid = compute_bins(self.Nf, self.Nt, self.tmax)

    def _initialize_gap_window(self):
        """Set up the gap window if `gap_kwargs` are provided."""
        gap_ranges = self.gap_kwargs.get("gap_ranges", None)
        gap_type = self.gap_kwargs.get("type", GapType.STITCH)
        if gap_ranges:
            self.gaps = GapWindow(
                self.time, tmax=self.tmax, gap_ranges=gap_ranges, type=gap_type
            )
        else:
            self.gaps = None

    @property
    def ND(self) -> int:
        return self.__ND

    @ND.setter
    def ND(self, n: int):
        """Ensure ND is a power of 2, raising an error with a suggestion if it is not."""
        n_pwr_2 = 2 ** int(np.log2(n))
        if n != n_pwr_2:
            suggestion = dict(dt=self.dt, tmax=n_pwr_2 * self.dt)
            current = dict(dt=self.dt, tmax=self.tmax)
            raise ValueError(
                f"ND must be a power of 2."
                f"Current settings:\n\t {current}\n"
                f"Suggested settings:\n\t {suggestion}"
            )
        self.__ND = n

    @property
    def psd_freqseries(self) -> FrequencySeries:
        """Generate the frequency series PSD if it hasn't been computed."""
        if not hasattr(self, "__psd_freqseries"):
            self.__psd_freqseries = CornishPowerSpectralDensity(self.freq)
        return self.__psd_freqseries

    @property
    def psd_wavelet(self) -> Wavelet:
        """Compute the wavelet form of the evolutionary PSD."""
        if not hasattr(self, "__psd_wavelet"):
            psd = self.psd_freqseries
            self.__psd_wavelet = evolutionary_psd_from_stationary_psd(
                psd.data, psd.freq, self.f_grid, self.t_grid, self.dt
            )
        return self.__psd_wavelet

    @property
    def psd(self) -> Wavelet:
        """Return the PSD for the analysis."""
        if not hasattr(self, "__psd"):
            p = self.psd_wavelet.copy()
            if self.gaps:
                p = self.gaps.apply_nan_gap_to_wavelet(p)
            self.__psd = p
        return self.__psd

    @property
    def ht(self) -> TimeSeries:
        """Generate the time series from the waveform generator if provided."""
        if not hasattr(self, "__ht"):
            self.__ht = (
                self.waveform_generator(*self.waveform_parameters, self.time)
                if self.waveform_generator
                else TimeSeries._EMPTY(self.ND, self.dt)
            )

        return self.__ht

    @property
    def noise_timeseries(self) -> TimeSeries:
        """Generate stationary noise time series if noise is enabled."""
        if not hasattr(self, "__noise_timeseries"):
            self.__noise_timeseries = (
                generate_stationary_noise(
                    ND=self.ND, dt=self.dt, psd=self.psd_freqseries
                )
                if self.noise
                else TimeSeries._EMPTY(self.ND, self.dt)
            )
        return self.__noise_timeseries

    @property
    def data_timeseries(self) -> TimeSeries:
        """Combine the signal and noise time series."""
        if not hasattr(self, "__data_timeseries"):
            self.__data_timeseries = self.ht + self.noise_timeseries
        return self.__data_timeseries

    @property
    def hf(self) -> FrequencySeries:
        """Convert time series to frequency series."""
        if not hasattr(self, "__hf"):
            self.__hf = self.ht.to_frequencyseries()
        return self.__hf

    @property
    def data_frequencyseries(self) -> FrequencySeries:
        """Convert data time series to frequency series."""
        if not hasattr(self, "__data_frequencyseries"):
            self.__data_frequencyseries = (
                self.data_timeseries.to_frequencyseries()
            )
        return self.__data_frequencyseries

    @property
    def hwavelet(self) -> Wavelet:
        """Compute wavelet transform of the time series."""
        if not hasattr(self, "__hwavelet"):
            self.__hwavelet = self.ht.to_wavelet(Nf=self.Nf)
        return self.__hwavelet

    @property
    def hwavelet_gapped(self) -> Wavelet:
        """Apply gap windowing to the wavelet-transformed time series."""
        if not hasattr(self, "__hwavelet_gapped"):
            self.__hwavelet_gapped = self.gaps.gap_n_transform_timeseries(
                self.ht, self.Nf, self.alpha, self.highpass_fmin
            )
        return self.__hwavelet_gapped

    @property
    def data_wavelet(self) -> Wavelet:
        """Apply gap windowing and high-pass filtering to data time series and compute wavelet."""
        if not hasattr(self, "__data_wavelet"):
            data_timeseries = self.data_timeseries
            if self.highpass_fmin > 0:
                data_timeseries = data_timeseries.highpass_filter(
                    fmin=self.highpass_fmin, tukey_window_alpha=self.alpha
                )
            self.__data_wavelet = (
                data_timeseries.to_wavelet(Nf=self.Nf)
                if not self.gaps
                else self.gaps.gap_n_transform_timeseries(
                    data_timeseries, self.Nf, self.alpha, self.highpass_fmin
                )
            ) 
        return self.__data_wavelet

    @property
    def summary_dict(self) -> Dict[str, float]:
        """Summary dictionary of analysis metrics, including signal-to-noise ratios (SNR)."""
        if not hasattr(self, "__summary_dict"):
            self.__summary_dict = dict(
                ht=self.ht,
                gaps=self.gaps,
                windowed=self.highpass_fmin > 0,
                noise=self.noise,
                **self.snr_dict,
            )
        return self.__summary_dict

    @property
    def summary(self) -> str:
        """Formatted summary string of analysis metrics."""
        return "\n".join([f"{k}: {v}" for k, v in self.summary_dict.items()])

    @property
    def snr_dict(self) -> Dict[str, float]:
        """Calculate various SNR values based on the analysis data."""
        if not hasattr(self, "__snr_dict"):
            self.__snr_dict = self._compute_snr_dict()
        return self.__snr_dict

    def _compute_snr_dict(self) -> Dict[str, float]:
        """Helper to calculate and return a dictionary of SNR values."""
        snrs = {
            "optimal_snr": self.hf.optimal_snr(self.psd_freqseries),
            "matched_filter_snr": self.hf.matched_filter_snr(
                self.data_frequencyseries, self.psd_freqseries
            ),
            "optimal_wavelet_snr": compute_snr(
                self.hwavelet, self.hwavelet, self.psd_wavelet
            ),
            "matched_filter_wavelet_snr": compute_snr(
                self.data_wavelet, self.hwavelet, self.psd_wavelet
            ),
            "optimal_data_wavelet_snr": compute_snr(
                self.hwavelet, self.hwavelet, self.psd
            ),
            "matched_filter_data_wavelet_snr": compute_snr(
                self.data_wavelet, self.hwavelet, self.psd
            ),
        }
        if self.gaps:
            snrs.update(
                {
                    "optimal_data_wavelet_snr": compute_snr(
                        self.hwavelet_gapped, self.hwavelet_gapped, self.psd
                    ),
                    "matched_filter_data_wavelet_snr": compute_snr(
                        self.data_wavelet, self.hwavelet_gapped, self.psd
                    ),
                }
            )

        for k, v in snrs.items():
            if np.isfinite(v):
                snrs[k] = np.round(v, 2)
        return snrs

    def plot_data(self, plotfn: str):
        """Plot data visualizations including SNR information, time series, wavelet, and PSDs."""
        fig, ax = plt.subplots(6, 1, figsize=(5, 8))
        ax[0].axis("off")
        ax[0].text(
            0.1, 0.5, self.summary, fontsize=8, verticalalignment="center"
        )

        self.hf.plot_periodogram(ax=ax[1], color="C0", alpha=1, lw=1)
        self.psd_freqseries.plot(ax=ax[1], color="k", alpha=1, lw=1)
        ax[1].set_xlim(left=self.highpass_fmin, right=self.freq[-1])
        ax[1].tick_params(
            axis="x",
            direction="in",
            labelbottom=False,
            top=True,
            labeltop=True,
        )

        self.ht.plot(ax=ax[2])
        kwgs = dict(show_colorbar=False, absolute=True, zscale="log")
        kwgs2 = dict(whiten_by=self.psd.data, **kwgs)
        self.data_wavelet.plot(ax=ax[3], label="Data\n", **kwgs2)
        self.hwavelet.plot(ax=ax[4], label="Model\n", **kwgs2)
        self.psd.plot(ax=ax[5], label="PSD\n", **kwgs)
        if self.gaps:
            if self.gaps.type == GapType.STITCH:
                chunks = self.gaps._chunk_timeseries(
                    self.ht, alpha=self.alpha, fmin=self.highpass_fmin
                )
                chunksf = [c.to_frequencyseries() for c in chunks]
                for i in range(len(chunks)):
                    chunksf[i].plot_periodogram(
                        ax=ax[1], color=f"C{i + 1}", alpha=0.5
                    )
                    chunks[i].plot(ax=ax[2], color=f"C{i + 1}", alpha=0.5)
            for a in ax[2:]:
                self.gaps.plot(ax=a)
        plt.subplots_adjust(hspace=0)
        plt.savefig(plotfn, bbox_inches="tight")

    def htemplate(self, *args, **kwargs) -> Wavelet:
        ht = self.waveform_generator(*args, **kwargs, t=self.time)
        if self.gaps is not None:
            hwavelet = self.gaps.gap_n_transform_timeseries(
                ht, self.Nf, self.alpha, self.highpass_fmin
            )
        else:
            if self.highpass_fmin > 0:
                ht = ht.highpass_filter(self.highpass_fmin, self.alpha)
            hwavelet = ht.to_wavelet(Nf=self.Nf)
        return hwavelet

    def lnl(self, *args) -> float:
        breakpoint()
        return compute_likelihood(
            self.data_wavelet, self.htemplate(*args), self.psd
        )
