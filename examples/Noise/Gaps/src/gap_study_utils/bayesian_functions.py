import numpy as np
from pywavelet.utils import compute_likelihood
from pywavelet.transforms.types import  Wavelet
from bilby.core.prior import Uniform, TruncatedGaussian, Gaussian, PriorDict
from .constants import A_RANGE, LN_F_RANGE, LN_FDOT_RANGE, A_TRUE, LN_F_TRUE, LN_FDOT_TRUE, A_SCALE, LN_F_SCALE, LN_FDOT_SCALE

from typing import List
from .wavelet_data_utils import gap_hwavelet_generator
from .analysis_data import AnalysisData

from .gap_funcs import GapWindow

PRIOR = PriorDict(dict(
    a=Uniform(*A_RANGE),
    ln_f=Uniform(*LN_F_RANGE),
    ln_fdot=Uniform(*LN_FDOT_RANGE)
))

CENTERED_PRIOR =  PriorDict(dict(
    a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE*0.1,  minimum=1e-25, maximum=1e-15),
    ln_f=Gaussian(mu=LN_F_TRUE, sigma=LN_F_SCALE*0.1),
    ln_fdot=Gaussian(mu=LN_FDOT_TRUE, sigma=LN_FDOT_SCALE*0.1)
))



def lnl(
        a: float,
        ln_f: float,
        ln_fdot: float,
        analysis_data: AnalysisData,
) -> float:
    htemplate = gap_hwavelet_generator(
        a=a, ln_f=ln_f, ln_fdot=ln_fdot,
        time=analysis_data.time,
        tmax=analysis_data.tmax,
        gap=analysis_data.gap,
        Nf=analysis_data.Nf,
        **analysis_data.waveform_kwgs
    )
    return compute_likelihood(analysis_data.hwavelet, htemplate, analysis_data.psd)



def log_prior(theta):
    a, ln_f, ln_fdot = theta
    _lnp = PRIOR.ln_prob(dict(a=a, ln_f=ln_f, ln_fdot=ln_fdot))
    if not np.isfinite(_lnp):
        return -np.inf
    else:
        return 0.0


def sample_prior(prior:PriorDict, n_samples=1)->np.ndarray:
    """Return (nsamp, ndim) array of samples"""
    return np.array(list(prior.sample(n_samples).values())).T



def log_posterior(theta:List[float], analysis_data:AnalysisData)->float:
    _lp = log_prior(theta)
    if not np.isfinite(_lp):
        return -np.inf
    else:
        _lnl = lnl(*theta, analysis_data=analysis_data)
        return _lp + _lnl


