import matplotlib.pyplot as plt
from typing import List, Dict, Union
import numpy as np

from pywavelet.utils import evolutionary_psd_from_stationary_psd, compute_snr
from pywavelet.transforms.types import Wavelet, TimeSeries, FrequencySeries


from .signal_utils import waveform_generator
from .noise_curves import CornishPowerSpectralDensity, generate_stationary_noise
from .wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap
from .gap_window import GapWindow

from .constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE,  NF, TMAX, GAP_RANGES

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
    gaps: List[GapWindow]
    trues: List[float]
    waveform_kwgs: Dict[str, Union[float, bool]]
    snrs: Dict[str, float]

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

    @property
    def alpha(self):
        return self.waveform_kwgs["alpha"]

    @property
    def filter(self)->bool:
        return self.waveform_kwgs["filter"]

    @property
    def fmin(self):
        return self.waveform_kwgs["fmin"]


    @classmethod
    def generate_data(
            cls,
            a_true: float = A_TRUE,
            ln_f_true: float = LN_F_TRUE,
            ln_fdot_true: float = LN_FDOT_TRUE,
            gap_ranges: List[float] = GAP_RANGES,
            Nf: int = NF,
            tmax: float = TMAX,
            noise: bool = False,
            alpha: float = 0.0,
            fmin: float = 7e-4,
            filter: bool = False,
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
        :param filter: Flag to apply a high-pass filter.
        :param plotfn: Filename to save the plot.

        :return: Wavelet data, Wavelet PSD, and GapWindow
        :type: Tuple[Wavelet, Wavelet, GapWindow]
         Tuple containing the wavelet-transformed data with gaps, the PSD with gaps, and the gap window.
        """
        f, fdot = np.exp(ln_f_true), np.exp(ln_fdot_true)
        params = [a_true, f, fdot]
        # ensure that the dt is small enough for given f and fdot
        dt = np.floor(0.4 / (2 * f))
        t = np.arange(0, tmax, dt)
        ht = waveform_generator(*params, t, tmax, alpha, )
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
            gaps = [GapWindow(data.time, gap_range, tmax=tmax) for gap_range in gap_ranges]
            print(f"Using Gap: {gaps}")
            data_wavelet = generate_wavelet_with_gap(
                gaps=gaps, ht=data, Nf=Nf, alpha=alpha, filter=filter, fmin=fmin
            )
            hw = generate_wavelet_with_gap(
                gaps=gaps, ht=ht, Nf=Nf, alpha=alpha, filter=filter, fmin=fmin
            )
            for g in gaps:
                psd_wavelet = g.apply_nan_gap_to_wavelet(psd_wavelet)
        else:
            print("------NO GAPS------")
            gaps = None
            data = data.highpass_filter(fmin=fmin, tukey_window_alpha=alpha)
            data_wavelet = data.to_wavelet(Nf=Nf)
            ht = ht.highpass_filter(fmin=fmin, tukey_window_alpha=alpha)
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
            dict(alpha=alpha, filter=filter, fmin=fmin),
            snrs
        )

        if plotfn:
            self.plot_data(plotfn)
        return self

    def plot_data(self, plotfn: str):

        hf = self.ht.to_frequencyseries()

        fig, ax = plt.subplots(5, 1,  figsize=(5, 8))

        # SNR info
        ax[0].axis("off")
        snr_text = "\n".join([f"{k}: {v:.2f}" for k, v in self.snrs.items()])
        ax[0].text(0.1, 0.5, snr_text, fontsize=8, verticalalignment="center")

        # timeseries + frequency series
        self.ht.plot(ax=ax[2], color="C0", label="Signal", alpha=0.9, lw=0.1)
        hf.plot_periodogram(ax=ax[1], color="C0", label="Signal", alpha=1, lw=1)
        ax[1].loglog(self.psd_freqseries.freq, self.psd_freqseries.data, color="k", label="PSD")
        if self.gaps is not None:
            chunks = chunk_timeseries(self.ht, self.gaps, windowing_alpha=self.alpha, filter=self.filter, fmin=self.fmin)
            chunksf = [c.to_frequencyseries() for c in chunks]
            for i in range(len(chunks)):
                chunks[i].plot(ax=ax[2], color=f"C{i + 1}", label=f"Chunk {i}")
                chunksf[i].plot_periodogram(ax=ax[1], color=f"C{i + 1}", label=f"Chunk {i}")
        # ax[1].legend(loc="upper right" )

        # Wavelet + psd
        self.wavelet_data.plot(ax=ax[3], show_colorbar=False)
        self.hwavelet.plot_trend(ax=ax[3])
        self.psd.plot(ax=ax[4], show_colorbar=False)

        plot_tmax = self.tmax * 1.1
        for a in ax[2:]:
            # if self.gaps is not None:
            #     a.axvspan(self.gap.gap_start, self.gap.gap_end, edgecolor='gray', hatch='/', zorder=10, fill=False)
            a.axvline(self.tmax, color="red", linestyle="--", label="Tmax")
            a.set_xlim(self.tmax-plot_tmax, plot_tmax)

        plt.savefig(plotfn, bbox_inches="tight")


