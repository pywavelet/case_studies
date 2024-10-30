from gap_study_utils.mcmc_runner import run_mcmc
from gap_study_utils.constants import START_GAP, END_GAP
import os

OUTDIR = "out_mcmc"
os.makedirs(OUTDIR, exist_ok=True)

if __name__ == '__main__':

    run_mcmc(
        gap_range=[START_GAP, END_GAP],
        n_iter=500,
        outdir=f"{OUTDIR}/gap",
    )
    run_mcmc(
        gap_range=None,
        n_iter=500,
        outdir=f"{OUTDIR}/no_gap",
    )
