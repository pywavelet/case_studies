import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import numpy as np
from numpy.ma.core import absolute

from pywavelet.utils import evolutionary_psd_from_stationary_psd, compute_snr, compute_likelihood
from pywavelet.transforms.types import Wavelet, TimeSeries, FrequencySeries

from .signal_utils import waveform_generator
from .noise_curves import CornishPowerSpectralDensity, generate_stationary_noise
from .gap_window import GapWindow

from .constants import F_TRUE, A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, NF, TMAX, GAP_RANGES, DT

from dataclasses import dataclass


@dataclass
class AnalysisData:
    """Packed data object with everything needed for the gap study."""
    ht: TimeSeries
    wavelet_data: Wavelet
    hwavelet: Wavelet
    tmax: float
    psd: Wavelet
    psd_freqseries: FrequencySeries
    gaps: GapWindow
    trues: List[float]
    snrs: Dict[str, float]
    alpha: float = 0.0
    highpass_fmin: float = 7e-4

    @property
    def dt(self):
        return self.ht.dt

    @property
    def ND(self):
        return self.ht.ND

    @property
    def Nf(self):
        return self.hwavelet.Nf

    @property
    def time(self):
        return self.ht.time


    @classmethod
    def generate_data(
            cls,
            a_true: float = A_TRUE,
            ln_f_true: float = LN_F_TRUE,
            ln_fdot_true: float = LN_FDOT_TRUE,
            gap_ranges: List[Tuple[float, float]] = GAP_RANGES,
            Nf: int = NF,
            tmax: float = TMAX,
            dt:float = DT,
            noise: bool = False,
            alpha: float = 0.0,
            highpass_fmin: float = 0,
            plotfn: str = "",
    ) -> "AnalysisData":
        """
        Generate data with gaps and corresponding PSD in the wavelet domain.

        :param a_true: Amplitude of the signal.
        :param ln_f_true: Natural logarithm of the frequency.
        :param ln_fdot_true: Natural logarithm of the frequency derivative.
        :param gap_ranges: [[Start, end]] times of the gaps (in seconds).
        :param Nf: Number of frequency bins.
        :param tmax: Maximum time for the signal (in seconds).
        :param noise_realisation: Flag to include noise realisation.
        :param seed_no: Seed number for random noise generation.
        :param alpha: Alpha parameter for the windowing function.
        :param highpass_fmin: Minimum frequency for highpass filter. if None, no filter is applied.
        :param plotfn: Filename to save the plot.

        :return: Wavelet data, Wavelet PSD, and GapWindow
        :type: Tuple[Wavelet, Wavelet, GapWindow]
         Tuple containing the wavelet-transformed data with gaps, the PSD with gaps, and the gap window.
        """
        f, fdot = np.exp(ln_f_true), np.exp(ln_fdot_true)
        params = [a_true, f, fdot]
        N = 2**int(np.floor(np.log2(tmax / dt)))
        tmax = N * dt
        t = np.arange(0, tmax, dt)

        print(f"Generating data with dt={dt:.2e}, N=2**{int(np.log2(N))}={N:,}")
        ht = waveform_generator(*params, t, tmax, alpha )
        hf = ht.to_frequencyseries()
        hw = ht.to_wavelet(Nf=Nf)
        psd = CornishPowerSpectralDensity(hf.freq)
        psd_wavelet = evolutionary_psd_from_stationary_psd(
            psd=psd.data, psd_f=psd.freq, f_grid=hw.freq,
            t_grid=hw.time, dt=hf.dt
        )

        data = ht.copy()
        data_wavelet = hw.copy()

        if noise:
            print("Cooking up some noise...")
            noise_t = generate_stationary_noise(ND=ht.ND, dt=ht.dt, psd=psd)
            data = noise_t + data
            data_wavelet = data.to_wavelet(Nf=Nf)

        optimal_snr = hf.optimal_snr(psd)
        optimal_wavelet_snr = compute_snr(hw, hw, psd_wavelet)
        matched_filter_snr = hf.matched_filter_snr(data.to_frequencyseries(), psd)
        matched_filter_wavelet_snr = compute_snr(data_wavelet, hw, psd_wavelet)

        print(f"Data: {ht}")
        print(f"Optimal freq-SNR: {optimal_snr:.2f}")
        print(f"Matched-filter freq-SNR: {matched_filter_snr:.2f}")
        print(f"Optimal WDM-SNR: {optimal_wavelet_snr:.2f}")
        print(f"Matched-filter WDM-SNR: {matched_filter_wavelet_snr:.2f}")

        # Gap data
        if gap_ranges is not None:
            print("------GAPPING------")
            gaps = GapWindow(data.time, gap_ranges, tmax=tmax)
            print(f"Using Gap: {gaps}")
            data_wavelet = gaps.gap_timeseries_chunk_transform_wdm_n_stitch(data, Nf, alpha, highpass_fmin)
            hw = gaps.gap_timeseries_chunk_transform_wdm_n_stitch(ht, Nf, alpha, highpass_fmin)
            psd_wavelet = gaps.apply_nan_gap_to_wavelet(psd_wavelet)
        else:
            print("------NO GAPS------")
            gaps = None
            if highpass_fmin>0:
                data = data.highpass_filter(fmin=highpass_fmin, tukey_window_alpha=alpha)
                ht = ht.highpass_filter(fmin=highpass_fmin, tukey_window_alpha=alpha)
            data_wavelet = data.to_wavelet(Nf=Nf)
            hw = ht.to_wavelet(Nf=Nf)
            psd_wavelet = psd_wavelet

        optimal_gapped_wdm_snr = compute_snr(hw, hw, psd_wavelet)
        matched_filter_gapped_wdm_snr = compute_snr(data_wavelet, hw, psd_wavelet)
        print(f"Optimal gapped WDM-SNR: {optimal_gapped_wdm_snr:.2f}")
        print(f"Matched-filter gapped WDM-SNR: {matched_filter_gapped_wdm_snr:.2f}")
        assert matched_filter_gapped_wdm_snr != 0, "SNR is 0... Something went wrong!"

        snrs = dict(
            optimal_freq=optimal_snr,
            matched_filter_freq=matched_filter_snr,
            optimal_wavelet=optimal_wavelet_snr,
            matched_filter_wavelet=matched_filter_wavelet_snr,
            optimal_gapped_wavelet=optimal_gapped_wdm_snr,
            matched_filter_gapped_wavelet=matched_filter_gapped_wdm_snr
        )

        self = cls(
            ht,
            data_wavelet,
            hw,
            tmax,
            psd_wavelet,
            psd,
            gaps,
            [a_true, ln_f_true, ln_fdot_true],
            snrs,
            alpha,
            highpass_fmin
        )

        if plotfn:
            self.plot_data(plotfn)
        return self

    def plot_data(self, plotfn: str):

        hf = self.ht.to_frequencyseries()

        fig, ax = plt.subplots(6, 1, figsize=(5, 8))

        # SNR info
        ax[0].axis("off")
        snr_text = "\n".join([f"{k}: {v:.2f}" for k, v in self.snrs.items()])
        ax[0].text(0.1, 0.5, snr_text, fontsize=8, verticalalignment="center")

        # timeseries + frequency series
        hf.plot_periodogram(ax=ax[1], color="C0", label="Signal", alpha=1, lw=1)
        ax[1].loglog(self.psd_freqseries.freq, self.psd_freqseries.data, color="k", label="PSD")
        ax[1].set_xlim(left=self.highpass_fmin, right=hf.freq[-1])
        ax[1].tick_params(axis="x", direction="in", labelbottom=False,top=True, labeltop=True)
        self.ht.plot(ax=ax[2], color="C0", label="Signal", alpha=0.9, lw=0.1)
        if self.gaps is not None:
            chunks = self.gaps.chunk_timeseries(self.ht, alpha=self.alpha, fmin=self.highpass_fmin)
            chunksf = [c.to_frequencyseries() for c in chunks]
            for i in range(len(chunks)):
                chunks[i].plot(ax=ax[2], color=f"C{i + 1}", label=f"Chunk {i}", alpha=0.5)
                chunksf[i].plot_periodogram(ax=ax[1], color=f"C{i + 1}", label=f"Chunk {i}", alpha=0.5)

        # Wavelet + psd
        self.wavelet_data.plot(ax=ax[3], show_colorbar=False, label="Whiten Data\n", whiten_by=self.psd.data, absolute=True,zscale="log")
        self.hwavelet.plot(ax=ax[4], show_colorbar=False, label="Signal\n", absolute=True,zscale="log")
        self.psd.plot(ax=ax[5], show_colorbar=False, label="PSD\n", absolute=True,zscale="log")

        # remove hspace between plots [2:]
        plt.subplots_adjust(hspace=0)
        for a in ax[2:]:
            a.axvline(self.tmax, color="red", linestyle="--", label="Tmax")
            a.set_xlim(0, self.tmax)
            if self.gaps:
                self.gaps.plot(ax=a, alpha=0.8, edgecolor='lightgray')
        for a in ax[3:]:
            a.set_ylim(bottom=self.highpass_fmin)
        plt.savefig(plotfn, bbox_inches="tight")

    def htemplate(self, a: float, ln_f: float, ln_fdot: float):
        f, fdot = np.exp(ln_f), np.exp(ln_fdot)
        ht = waveform_generator(a, f, fdot, self.time, self.tmax, self.alpha)
        if self.gaps is not None:
            hwavelet = self.gaps.gap_timeseries_chunk_transform_wdm_n_stitch(
                ht, self.Nf, self.alpha, self.highpass_fmin
            )
        else:
            if self.highpass_fmin>0:
                ht = ht.highpass_filter(self.highpass_fmin, self.alpha)
            hwavelet = ht.to_wavelet(Nf=self.Nf)
        return hwavelet

    def lnl(self, a: float, ln_f: float, ln_fdot: float) -> float:
        return compute_likelihood(
            self.wavelet_data,
            self.htemplate(a, ln_f, ln_fdot),
            self.psd
        )
