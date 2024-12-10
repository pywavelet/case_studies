import os

from gap_study_utils.constants import F_TRUE
from gap_study_utils.mcmc_runner import run_mcmc

OUTDIR = "out_mcmc"
os.makedirs(OUTDIR, exist_ok=True)
NITER = 500
DT = 20
TMAX = 327_680
GAPS = [
    [TMAX * 0.22, TMAX * 0.3],
    [TMAX * 0.52, TMAX * 0.63],
]

common_kwgs = dict(
    n_iter=NITER,
    alpha=0.0,
    highpass_fmin=0.0,# * F_TRUE / 4,
    dt=DT,
    tmax=TMAX,
)


if __name__ == "__main__":
    # run_mcmc(
    #     gap_ranges=GAPS,
    #     noise_realisation=True,
    #     outdir=f"{OUTDIR}/gap+noise",
    #     **common_kwgs
    # )
    run_mcmc(
        gap_ranges=None,
        noise_realisation=True,
        outdir=f"{OUTDIR}/noise",
        **common_kwgs
    )
    # run_mcmc(
    #     gap_ranges=None,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/basic",
    #     **common_kwgs,
    # )
    # run_mcmc(
    #     gap_ranges=GAPS,
    #     noise_realisation=False,
    #     outdir=f"{OUTDIR}/gap",
    #     **common_kwgs
    # )
