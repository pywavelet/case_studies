from gap_study_utils.noise_curves import generate_stationary_noise
import matplotlib.pyplot as plt
import numpy as np


def test_generate_stationary_noise(test_data, plot_dir):
    noise_f = generate_stationary_noise(
        ND=test_data.ND,
        dt=test_data.dt,
        psd=test_data.psd_freqseries,
        time_domain=False
    )
    noise_t = generate_stationary_noise(
        ND=test_data.ND,
        dt=test_data.dt,
        psd=test_data.psd_freqseries,
        time_domain=True
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    noise_f.plot_periodogram(ax=axes[0])
    test_data.psd_freqseries.plot_periodogram(ax=axes[0])
    noise_t.plot(ax=axes[1])
    fig.savefig(f"{plot_dir}/stationary_noise.png")


    for noise in [noise_f, noise_t]:
        # assert non-zero, non-nan values for entire time series
        assert np.all(~np.isnan(noise.data)), f"Noise {noise} has NaNs"
        assert np.all(noise.data != 0), f"Noise {noise} is all zeros"


