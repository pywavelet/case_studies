import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

from pywavelet.utils import evolutionary_psd_from_stationary_psd, compute_snr
from pywavelet.transforms.types import Wavelet, TimeSeries, FrequencySeries
from pywavelet.transforms import from_freq_to_wavelet

from .signal_utils import compute_snr_freq, generate_padded_signal
from .noise_curves import CornishPowerSpectralDensity, generate_stationary_noise
from .wavelet_data_utils import chunk_timeseries, generate_wavelet_with_gap
from .gap_funcs import GapWindow

from .constants import A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, START_GAP, END_GAP, NF, TMAX

from dataclasses import dataclass


@dataclass
class AnalysisData:
    """Packed data object with everything needed for the gap study."""
    ht: TimeSeries
    hwavelet: Wavelet
    tmax: float
    psd: Wavelet
    psd_freqseries: FrequencySeries
    gap: GapWindow
    trues: List[float]
    waveform_kwgs: Dict[str, float]

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
    def filter(self):
        return self.waveform_kwgs["filter"]


    @classmethod
    def generate_data(
            cls,
            a_true: float = A_TRUE,
            ln_f_true: float = LN_F_TRUE,
            ln_fdot_true: float = LN_FDOT_TRUE,
            gap_range: List[float] = [START_GAP, END_GAP],
            Nf: int = NF,
            tmax: float = TMAX,
            noise: bool = False,
            alpha: float = 0.0,
            filter: bool = False,
            plotfn: str = "",
    ) -> "AnalysisData":
        """
        Generate data with gaps and corresponding PSD in the wavelet domain.

        :param a_true: Amplitude of the signal.
        :param ln_f_true: Natural logarithm of the frequency.
        :param ln_fdot_true: Natural logarithm of the frequency derivative.
        :param gap_range: [Start, end] time of the gap (in seconds).
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

        ht, hf = generate_padded_signal(a_true, ln_f_true, ln_fdot_true, tmax)
        h_wavelet = from_freq_to_wavelet(hf, Nf=Nf)
        psd = CornishPowerSpectralDensity(hf.freq)
        psd_wavelet = evolutionary_psd_from_stationary_psd(
            psd=psd.data, psd_f=psd.freq, f_grid=h_wavelet.freq,
            t_grid=h_wavelet.time, dt=hf.dt
        )
        print(f"SNR (hf, no gaps): {compute_snr_freq(hf.data, psd.data, hf.dt, hf.ND)}")
        print(f"SNR (hw, no gaps/noise): {compute_snr(h_wavelet, psd_wavelet)}")

        if noise:
            # TODO: this isnt working for me...
            noise_t = generate_stationary_noise(ND=ht.ND, dt=ht.dt, psd=psd)
            data = noise_t + ht
        else:
            data = ht

        # Gap data
        if gap_range is not None:
            gap = GapWindow(data.time, gap_range, tmax=tmax)
            print(f"Using Gap: {gap}")
            data_wavelet = generate_wavelet_with_gap(
                gap=gap, ht=data, Nf=Nf, alpha=alpha, filter=filter
            )
            psd_wavelet = gap.apply_nan_gap_to_wavelet(psd_wavelet)
        else:
            gap = None
            data_wavelet = from_freq_to_wavelet(data.to_frequencyseries(), Nf)
            psd_wavelet = psd_wavelet

        snr_data = compute_snr(data_wavelet, psd_wavelet)

        print(f"SNR (analysis wavelet data): {snr_data}")
        assert snr_data != 0, "SNR is 0... Something went wrong!"

        self = cls(
            ht,
            data_wavelet,
            tmax,
            psd_wavelet,
            psd,
            gap,
            [a_true, ln_f_true, ln_fdot_true],
            dict(alpha=alpha, filter=filter)
        )

        if plotfn:
            self.plot_data(plotfn)

        return self

    def plot_data(self, plotfn: str):
        fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 5))
        self.ht.plot(ax=ax[0], color="C0", label="Signal", alpha=0.5, lw=0.1)
        if self.gap is not None:
            chunks = chunk_timeseries(self.ht, self.gap)
            for i, chunk in enumerate(chunks):
                chunk.plot(ax=ax[0], color=f"C{i + 1}", label=f"Chunk {i}")
        ax[0].legend(loc="upper right")
        self.hwavelet.plot(ax=ax[1], show_colorbar=False)
        self.psd.plot(ax=ax[2], show_colorbar=False)
        for a in ax:
            if self.gap is not None:
                a.axvspan(self.gap.gap_start, self.gap.gap_end, edgecolor='gray', hatch='/', zorder=10, fill=False)
            a.axvline(self.tmax, color="red", linestyle="--", label="Tmax")
        ax[0].set_xlim(0, self.tmax * 1.1)
        plt.subplots_adjust(hspace=0)
        plt.savefig(plotfn, bbox_inches="tight")

