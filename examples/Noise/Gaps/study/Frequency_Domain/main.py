import check_matrices
import estm_gap_cov
import gated_noise
from rich.progress import track

a_true = 1e-21
f_true = 3e-3
fdot_true = 1e-8
tmax = 10 * 60 * 60  # Final time


for tdi in track(
    ["Cornish", "TDI1", "TDI2"], description="Processing gated noise..."
):
    gated_noise.main(
        a_true=a_true,
        f_true=f_true,
        fdot_true=fdot_true,
        TDI=tdi,
        tmax=tmax,
    )

for tdi in track(
    ["TDI1", "TDI2"], description="Processing estim cov noise..."
):
    estm_gap_cov.main(
        a_true=a_true,
        f_true=f_true,
        fdot_true=fdot_true,
        TDI=tdi,
        tmax=tmax,
    )

check_matrices.main()
