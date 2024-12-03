import matplotlib.pyplot as plt
import numpy as np
from pywavelet.utils import evolutionary_psd_from_stationary_psd

from gap_study_utils.noise_curves import (
    CornishPowerSpectralDensity,
    generate_stationary_noise,
)

S_IN_DAY = 60 * 60 * 24


def test_generate_stationary_noise(plot_dir):
    rough_duration = 0.25 * S_IN_DAY
    fs = 10
    dt = 1 / fs

    # N = closest power of 2 to duration * fs
    N = 2 ** int(np.ceil(np.log2(rough_duration * fs)))
    duration = N * dt
    label = f"N = {N:,}, T = {duration/ S_IN_DAY:.2f} days"
    print(f"Noise {label}")

    freq = np.fft.rfftfreq(N, dt)
    psd = CornishPowerSpectralDensity(freq)
    noise_f = generate_stationary_noise(
        ND=N, dt=dt, psd=psd, time_domain=False
    )
    noise_t = generate_stationary_noise(
        ND=N,
        dt=dt,
        psd=psd,
        time_domain=True,
    )
    filtered_time = noise_t.highpass_filter(
        fmin=10**-3, tukey_window_alpha=0.1
    )
    filtered_freq = filtered_time.to_frequencyseries()

    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    fig.suptitle(label)
    axes[0].loglog(psd.freq, psd.data, color="black", label="PSD")
    noise_f.plot_periodogram(ax=axes[0], color="gray", label="Noise")
    noise_t.plot(ax=axes[1], color="gray")
    filtered_freq.plot_periodogram(
        ax=axes[0],
        color="tab:red",
        alpha=0.5,
        label="Filtered Noise (f > 10^-3 Hz)",
    )
    filtered_time.plot(ax=axes[1], color="tab:red", alpha=0.5)
    axes[0].legend()
    w = filtered_freq.to_wavelet(Nf=64)
    w_psd = evolutionary_psd_from_stationary_psd(
        psd.data, psd.freq, w.freq, w.time, dt=psd.dt
    )
    w.plot(
        ax=axes[2],
        show_colorbar=False,
        label="Filterd Wavelet\n",
        whiten_by=w_psd.data,
    )
    axes[0].tick_params(axis="x", which="both", top=True)
    plt.subplots_adjust(hspace=0)
    fig.savefig(f"{plot_dir}/stationary_noise.png")

    for noise in [noise_f, noise_t]:
        # assert non-zero, non-nan values for entire time series
        assert np.all(~np.isnan(noise.data)), f"Noise {noise} has NaNs"
        assert np.all(noise.data != 0), f"Noise {noise} is all zeros"
