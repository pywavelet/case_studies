import os
import sys
import numpy as np
from scipy.signal.windows import tukey
from tqdm import tqdm


from pywavelet.data import Data
from pywavelet.psd import evolutionary_psd_from_stationary_psd
from pywavelet.transforms.types import FrequencySeries, TimeSeries
from pywavelet.transforms.to_wavelets import (from_freq_to_wavelet, from_time_to_wavelet)
from pywavelet.transforms.from_wavelets import (from_wavelet_to_time)
from pywavelet.utils.lisa import waveform,  zero_pad
from pywavelet.plotting import plot_wavelet_grid
import matplotlib.pyplot as plt

sys.path.append("../../Frequency_Domain/")
from noise_curves import noise_PSD_AE, CornishPowerSpectralDensity
from wavelet_data_utils import stitch_together_data_wavelet, bandpass_data

# np.random.seed(1234)

ONE_HOUR = 60*60
def gap_routine_nan(t, start_window, end_window, delta_t = 10):

    start_window *= ONE_HOUR          # Define start of gap
    end_window *= ONE_HOUR          # Define end of gap

    N = len(t) 

    nan_window = []  # Initialise with empty vector
    j=0  
    for i in range(0,N):   # loop index i through length of t
        if t[i] > (start_window) and (t[i] < end_window):  # if t within gap segment
            nan_window.append(np.nan)  # add nan value (no data)
            j+=1  # incremement 
        else:                   # if t not within the gap segment
            nan_window.append(1)  # Just add a one.
            j=0
        
    return nan_window

def zero_pad(data):
    """
    This function takes in a vector and zero pads it so it is a power of two.
    We do this for the O(Nlog_{2}N) cost when we work in the frequency domain.
    """
    N = len(data)
    pow_2 = np.ceil(np.log2(N))
    return np.pad(data, (0, int((2**pow_2) - N)), "constant")

def inner_prod(sig1_f, sig2_f, PSD, delta_t, N_t):
    # Compute inner product. Useful for likelihood calculations and SNRs.
    return (4 * delta_t / N_t) * np.real(
        sum(np.conjugate(sig1_f) * sig2_f / PSD)
    )


def waveform(a, f, fdot, t, eps=0):
    """
    This is a function. It takes in a value of the amplitude $a$, frequency $f$ and frequency derivative $\dot{f}
    and a time vector $t$ and spits out whatever is in the return function. Modify amplitude to improve SNR.
    Modify frequency range to also affect SNR but also to see if frequencies of the signal are important
    for the windowing method. We aim to estimate the parameters $a$, $f$ and $\dot{f}$.
    """

    return a * (np.sin((2 * np.pi) * (f * t + 0.5 * fdot * t**2)))


a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8

# TDI = "TDI1"
TDI = "Cornish"
# if TDI == "Cornish": a_true = 1e-19

start_gap = 4
end_gap = 6

# Nf = 32 #works
Nf = 16 

tmax = 10 * 60 * 60  # Final time
fs = 2 * f_true  # Sampling rate
delta_t = np.floor(
    0.4 / fs
)  # Sampling interval -- largely oversampling here.

t = np.arange(
    0, tmax, delta_t
)  # Form time vector from t0 = 0 to t_{n-1} = tmax. Length N [include zero]

N = int(
    2 ** (np.ceil(np.log2(len(t))))
)  # Round length of time series to a power of two.
# Length of time series
h_t = waveform(a_true, f_true, fdot_true, t)
taper_signal = tukey(len(h_t), alpha = 0.0)
h_t_pad = zero_pad(h_t*taper_signal)

t_pad = np.arange(0,len(h_t_pad)*delta_t, delta_t)

h_true_f = np.fft.rfft(h_t_pad)
freq = np.fft.rfftfreq(N, delta_t); freq[0] = freq[1]
if TDI == "TDI1" or TDI == "TDI2":
    PSD = noise_PSD_AE(freq, TDI = TDI)
else:
    PSD = CornishPowerSpectralDensity(freq)

# freq  = np.arange(1e-5, 5e-1, 1e-7)
# PSD_TDI1 = noise_PSD_AE(freq,TDI = "TDI1")
# PSD_TDI2 = noise_PSD_AE(freq,TDI = "TDI2")
# PSD_Cornish = CornishPowerSpectralDensity(freq)

# plt.loglog(freq, PSD_TDI1, label = "TDI1")
# plt.loglog(freq, PSD_TDI2, label = "TDI2")
# plt.loglog(freq, PSD_Cornish, label = "Cornish")
# plt.legend()

# plt.title("Comparison, PSDs")
# plt.xlabel(r" Frequency [Hz]")
# plt.ylabel(r"Magnitude (seconds)")
# plt.grid()
# plt.show()
SNR2 = inner_prod(
    h_true_f, h_true_f, PSD, delta_t, N
)  # Compute optimal matched filtering SNR
print("SNR of source", np.sqrt(SNR2))

#  =============== WAVELET DOMAIN ========================

signal_f_series = FrequencySeries(data=h_true_f, freq=freq)
psd = FrequencySeries(data = PSD, freq = freq)

Nt = N//Nf
print("Length of signal: ",N)
print("Wavelet bins in frequency= ",Nf)
print("Wavelet bins in time =  ",Nt)

kwgs = dict(
    Nf=Nf,
)
h_wavelet = Data.from_frequencyseries(signal_f_series, **kwgs).wavelet
psd_wavelet = evolutionary_psd_from_stationary_psd(
                                                    psd=psd.data,
                                                    psd_f=psd.freq,
                                                    f_grid=h_wavelet.freq,
                                                    t_grid=h_wavelet.time,
                                                    dt=delta_t,
                                                )

Wavelet_Matrix = psd_wavelet.data

SNR2_wavelet = np.nansum((h_wavelet*h_wavelet) / psd_wavelet)
print("SNR in wavelet domain is", SNR2_wavelet**(1/2))

variance_noise_f = N * PSD/(4*delta_t)   # Compute variance in frequency domain (pos freq)

# Gaps in the frequency domain. 
w_t = gap_routine_nan(t_pad, start_window = start_gap, end_window = end_gap, delta_t = delta_t)

h_pad_w = w_t * h_t_pad 

plt.plot(t_pad/60/60, h_pad_w);
plt.xlabel(r'Time t [Hrs]')
plt.ylabel(r'Signal')
plt.title('Gaps')
plt.grid()
plt.savefig("../plots/waveform_nan.pdf",bbox_inches = "tight")
plt.clf()
h_approx_stitched_data, mask, template_window = stitch_together_data_wavelet(w_t, t_pad, h_t_pad, Nf, 
                                                                            delta_t, start_gap, end_gap, 
                                                                            windowing = False, alpha = 0.0, 
                                                                            filter = True)

# ===================== Old data set, force to have nans ===========================

Wavelet_Matrix_with_nans = Wavelet_Matrix.copy()
Wavelet_Matrix_with_nans[:,~mask] = np.nan
SNR2_original_data_gaps = np.nansum( h_approx_stitched_data * h_approx_stitched_data / Wavelet_Matrix_with_nans) 
SNR2_stitched_data_gaps = np.nansum( h_approx_stitched_data * h_approx_stitched_data / Wavelet_Matrix_with_nans) 

print("Using original data stream with gaps, we find SNR = ", np.sqrt(SNR2_original_data_gaps))
print("Using stitched together data stream with gaps, we find SNR = ", np.sqrt(SNR2_stitched_data_gaps))

# Can I do the same thing with noise? 

# np.random.seed(123)

noise_f_iter = np.random.normal(0,np.sqrt(variance_noise_f))  + 1j * np.random.normal(0,np.sqrt(variance_noise_f)) 
noise_f_iter[0] = np.sqrt(2)*noise_f_iter[0].real
noise_f_iter[-1] = np.sqrt(2)*noise_f_iter[-1].real
noise_t = np.fft.irfft(noise_f_iter)

noise_wavelet_stitched, _, _ = stitch_together_data_wavelet(w_t, t_pad, noise_t, Nf, delta_t, start_gap, end_gap, windowing = False , alpha = 0.0, filter = True)

data_set_wavelet_stitched = h_approx_stitched_data + 0*noise_wavelet_stitched 

# Let's try to estimate likelihood by generating a load of signals, gapping them and computing the likelihood 

precision = a_true / SNR2_original_data_gaps**(1/2)
n = 5
a_range = np.linspace(a_true - n*precision, a_true + n*precision, 200)

if a_true not in a_range:
    a_range = np.sort(np.insert(a_range, 0, a_true))

llike_vec = []

f_min = 9e-4
# ===================== Old data set, force to have nans ===========================
h_approx_stitched_data, mask, template_window = stitch_together_data_wavelet(w_t, t_pad, h_t_pad, Nf, 
                                                                            delta_t, start_gap, end_gap, 
                                                                            windowing = False, alpha = 0.0, 
                                                                            filter = True)

for a_val in a_range:

    h_prop = waveform(a_val, f_true, fdot_true, t)
    # h_prop_pad = template_window * zero_pad(h_prop)
    # h_prop_filter = bandpass_data(h_prop, f_min, 1/delta_t, bandpassing_flag = True, order = 4)
    
    # h_prop_pad = zero_pad(h_prop_filter)
    # h_prop_f = np.fft.rfft(h_prop_pad)
    # h_prop_f_FreqSeries = FrequencySeries(h_prop_f, freq=freq)
    # h_prop_wavelet = from_freq_to_wavelet(h_prop_f_FreqSeries, Nf = Nf)

    # taper_signal = tukey(len(h_prop), alpha = 0.0)
    h_prop_pad = zero_pad(taper_signal)
    h_prop_wavelet,_,_ = stitch_together_data_wavelet(w_t, t_pad, h_prop_pad, Nf, delta_t, start_gap, end_gap, windowing = False, alpha = 0.0, filter = True)

    llike_val = -0.5 * np.nansum ( ((h_approx_stitched_data - h_prop_wavelet)**2) / Wavelet_Matrix_with_nans) 
    llike_vec.append(llike_val)


llike_vec_array = np.array(llike_vec)
plt.plot(a_range, np.exp(llike_vec_array), '*')
plt.plot(a_range, np.exp(llike_vec_array))
plt.axvline(x = a_true, c = 'red', linestyle = '--', label = "truth")
plt.legend()
plt.xlabel(r'Amplitude values')
plt.ylabel(r'Likelihood')
# plt.yticks([])
if TDI == "TDI1" or TDI == "TDI2":
    plt.title(r'Likelihood -- With {} noise and Nf = {}'.format(TDI, Nf))
else:
    plt.title(r'Likelihood -- With {} noise and Nf = {}'.format("Cornish", Nf))
plt.show()
plt.clf()
# Extra processing

# h_prop = waveform(a_true, f_true, fdot_true, t)
# h_prop_pad = zero_pad(h_prop)

# h_prop_wavelet,mask,_ = stitch_together_data_wavelet(w_t, t, h_prop_pad, Nf, delta_t, start_gap, end_gap)

# llike_val = -0.5 * np.nansum ( ((noise_wavelet_stitched)**2) / Wavelet_Matrix_with_nans) 

# print("log_like value for noise = ", llike_val)
# print("Number of data points in time, N = ", N/2)

# Generate true signal with mask given by the gaps
breakpoint()
h_prop = waveform(a_val, f_true, fdot_true, t_pad)
h_prop_filter = bandpass_data(h_prop, f_min, 1/delta_t, bandpassing_flag = False, order = 4)

h_prop_pad = zero_pad(h_prop_filter)
h_prop_f = np.fft.rfft(h_prop_pad)
h_prop_f_FreqSeries = FrequencySeries(h_prop_f, freq=freq)
h_prop_wavelet = from_freq_to_wavelet(h_prop_f_FreqSeries, Nf = Nf)

h_prop_wavelet_mask = h_prop_wavelet.copy()
h_prop_wavelet_mask[:,~mask] = np.nan
# Plot the data

# Plot wavelet grid


h_approx_stitched_data, mask, template_window = stitch_together_data_wavelet(w_t, t, h_prop_pad, Nf, delta_t, start_gap, end_gap, windowing = True, alpha = 0.4, filter = True)

stitched_matrix = np.nan_to_num(h_approx_stitched_data, nan=0.0)
full_template_w_gap = np.nan_to_num(h_prop_wavelet_mask.data, nan=0.0)
fig, ax = plt.subplots(1, 2, figsize=(16, 7))
plot_wavelet_grid(
    stitched_matrix,
    time_grid=h_prop_wavelet.time,
    freq_grid=h_prop_wavelet.freq,
    ax=ax[0],
    zscale="linear",
    freq_scale="linear",
    absolute=False
    # freq_range=[0, 300]
)

ax[0].set_xlabel('Wavelet time bins')  # Custom x-label for the first subplot
ax[0].set_ylabel('Wavelet Frequency bins')  # Custom y-label for the first subplot
ax[0].set_title('Stitched together data set')  # Custom title for the first subplot
# Broken

fig = plot_wavelet_grid(
    full_template_w_gap,
    time_grid=h_prop_wavelet.time,
    freq_grid=h_prop_wavelet.freq,
    ax=ax[1],
    zscale="linear",
    freq_scale="linear",
    absolute=False
    # freq_range=[0, 300]
)

ax[1].set_xlabel('Wavelet time bins')  # Custom x-label for the first subplot
ax[1].set_ylabel('Wavelet Frequency bins')  # Custom y-label for the first subplot
ax[1].set_title('generated template with mask')  # Custom title for the first subplot
plt.show()
