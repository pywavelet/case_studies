from sympy.abc import alpha

from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import GAP_RANGES, F_TRUE
import os

OUTDIR = "out_mcmc"
os.makedirs(OUTDIR, exist_ok=True)
NITER = 1000
DT = 20
common_kwgs = dict(
    n_iter=NITER,
    alpha=0.0,
    highpass_fmin=F_TRUE / 4,
    dt=DT,
)


if __name__ == '__main__':
    run_mcmc(
        gap_ranges=GAP_RANGES,
        noise_realisation=True,
        outdir=f"{OUTDIR}/noise_gap",
        **common_kwgs
    )
    run_mcmc(
        gap_ranges=None,
        noise_realisation=True,
        outdir=f"{OUTDIR}/noise",
        **common_kwgs
    )
    run_mcmc(
        gap_ranges=None,
        noise_realisation=True,
        outdir=f"{OUTDIR}/no_gap_no_noise",
        **common_kwgs
    )
    run_mcmc(
        gap_ranges=GAP_RANGES,
        noise_realisation=False,
        outdir=f"{OUTDIR}/gap_no_noise",
        **common_kwgs
    )
