# Initial values plus sampling properties


from numba import njit, prange
from pywavelet.transforms.types import TimeSeries, Wavelet
from scipy.signal.windows import tukey
import numpy as np

ONE_HOUR = 60*60
def gap_routine(t:np.ndarray, start_window:float, end_window:float, lobe_length = 3, delta_t = 10):

    start_window *= ONE_HOUR          # Define gap_start of gap
    end_window *= ONE_HOUR          # Define gap_end of gap
    lobe_length *= ONE_HOUR          # Define length of cosine lobes

    window_length = int(np.ceil(((end_window+lobe_length) - 
                                (start_window - lobe_length))/delta_t))  # Construct of length of window 
                                                                        # throughout the gap
    alpha_gaps = 2*lobe_length/(delta_t*window_length)      # Construct alpha (windowing parameter)
                                                    # so that we window BEFORE the gap takes place.
    
    window = tukey(window_length,alpha_gaps)   # Construct window

    new_window = []  # Initialise with empty vector
    j=0  
    for i in range(0,len(t)):   # loop index i through length of t
        if t[i] > (start_window - lobe_length) and (t[i] < end_window + lobe_length):  # if t within gap segment
            new_window.append(1 - window[j])  # add windowing function to vector of ones.
            j+=1  # incremement 
        else:                   # if t not within the gap segment
            new_window.append(1)  # Just add a one.
            j=0
        

    alpha_full = 0.2
    total_window = tukey(len(new_window), alpha = alpha_full)

    new_window *= total_window
    return new_window

@njit()
def get_Cov(Cov_Matrix, w_fft, w_fft_star, delta_f, PSD):
    """
        Compute the covariance matrix for a given set of parameters.

        Parameters
        ----------
        Cov_Matrix : numpy.ndarray (Complex128)
            The initial covariance matrix to be updated.
        w_fft : numpy.ndarray
            Two-sided fourier transform of window function.
        w_fft_star : numpy.ndarray
            An array representing the conjugate transpose of the weights.
        delta_f : float
            The frequency increment.
        PSD : numpy.ndarray
            The Power Spectral Density array.

        Returns
        -------
        numpy.ndarray
            The updated covariance matrix.

        Notes
        -----
        This function performs the following steps:
        1. Computes a prefactor using `delta_f`.
        2. Iterates over the upper triangle of the matrix to update its elements.
        3. Enforces Hermitian symmetry by adjusting the diagonal and adding the conjugate transpose.
    
        The function is optimized using Numba's `njit` and `prange` for parallel execution.
        """ 
    pre_fac = 0.5 * delta_f
    n_t = len(w_fft) # w_fft is two sided transform, so same dimensions as N
    for i in prange(0, n_t//2 +1 ): # For each column
        if i % 50 == 0:
            print(i)
        for j in range(i,n_t//2 + 1 ):  # For each row
             Cov_Matrix[i,j] = pre_fac  * np.sum(PSD * np.roll(w_fft, i)*np.roll(w_fft_star, j))
    
    diagonal_elements = np.diag(np.diag(Cov_Matrix)) # Take real part of matrix
    diagonal_elements_real = 0.5*np.real(diagonal_elements)
    Cov_Matrix = Cov_Matrix - diagonal_elements + diagonal_elements_real 
    Cov_Matrix = Cov_Matrix + np.conjugate(Cov_Matrix.T)
    return Cov_Matrix


def regularise_matrix(Cov_Matrix, window, tol = 0.01):
    """
    Inputs: Cov_Matrix : Noise covariance matrix
            window : window function
            tol (arg): Essentially fraction of singular values to ignore in window

    Outputs: Cov_Matrix_reg_inv : Regularised inverse
    """

    U,S,Vh = np.linalg.svd(Cov_Matrix)      # Compute SVD
    N_remove = len(np.argwhere(window <= tol))  # Number of singular values to remove
    N_retain = len(S) - N_remove               # Compute number of singular values to retain
    S_inv = S**-1                              # Compute inverse of singular matrix. 
    
    S_inv_regular = []
    for i in range(0,len(S)):
        if i >= N_retain: 
            S_inv_regular.append(0)           # Set infinite singular values to zero. 
        else:
            S_inv_regular.append(S_inv[i])
    Cov_Matrix_reg_inv = Vh.T.conj() @ np.diag(S_inv_regular) @ U.conj().T
    np.fill_diagonal(Cov_Matrix_reg_inv, np.real(np.diag(Cov_Matrix_reg_inv))) # Force diagonal to be real. 
    return Cov_Matrix_reg_inv







class GapWindow:
    def __init__(self, t:np.ndarray, start:float, end:float):
        self.gap_start = start
        self.gap_end = end
        self.nan_mask = self.__generate_nan_mask(t, start, end)
        self.t = t
        # first idx od nan
        self.start_idx = np.argmax(self.nan_mask)
        # last idx of nanx
        self.end_idx = len(self.nan_mask) - np.argmax(self.nan_mask[::-1]) -1


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
        return self.num_nans/len(self.nan_mask)

    def __repr__(self):
        return f"GapWindow({self.num_nans:,}/{len(self.nan_mask):,} NaNs)"

    @staticmethod
    def __generate_nan_mask(t:np.ndarray, start_window:float, end_window:float):
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


    def apply_window(self, timeseries)->TimeSeries:
        """
        Apply the gap window to a given time series.

        Parameters:
        - timeseries: Time series to apply the gap window.

        Returns:
        - TimeSeries:
            Time series with NaN values in the gap window.

        """
        return TimeSeries(timeseries.data*self.nan_mask, self.t)

    def apply_nan_gap_to_wavelet(self, w:Wavelet):
        """
        Apply the gap window to a given wavelet.

        Parameters:
        - w: Wavelet to apply the gap window.

        Returns:
        - Wavelet:
            Wavelet with NaN values in the gap window.

        """
        data, t = w.data, w.time
        nan_mask = self.get_nan_mask_for_timeseries(t)
        time_mask = self.inside_timeseries(t)
        data[:, nan_mask] = np.nan
        return Wavelet(data[:, time_mask], t[time_mask], w.freq)


    def get_nan_mask_for_timeseries(self,t):
        """Gets a mask for the given time series -- True for NaNs, False for valid data."""
        return ~self.valid_t(t)

    def inside_gap(self, t:float):
        # if t inside [gap_start, gap_end], True
        # else False
        return (self.gap_start <= t) & (t <= self.gap_end)

    def inside_timeseries(self, t:np.ndarray):
        return (self.t[0] <= t) & (t <= self.t[-1])

    def valid_t(self, t:float):
        # if t inside timeseries and outside gap, True
        return self.inside_timeseries(t) &~ self.inside_gap(t)
