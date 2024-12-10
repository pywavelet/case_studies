import os

import numpy as np
from bilby.core.prior import Gaussian, PriorDict, TruncatedGaussian, Uniform

ONE_HOUR = 60 * 60
ONE_DAY = 24 * ONE_HOUR
A_TRUE = 1e-18
F_TRUE = 3e-3
FDOT_TRUE = 1e-8

LN_F_TRUE = np.log(F_TRUE)
LN_FDOT_TRUE = np.log(FDOT_TRUE)

TRUES = [A_TRUE, LN_F_TRUE, LN_FDOT_TRUE]

NF = 64
TMAX = 4 * ONE_DAY
# START_GAP = TMAX  * 0.4 # OLD
START_GAP = TMAX * 0.75  # Get the two segments the same size!
END_GAP = START_GAP + 6 * ONE_HOUR

A_RANGE = [1e-22, 3e-21]
F_RANGE = [F_TRUE - 6e-7, F_TRUE + 6e-7]
FDOT_RANGE = [FDOT_TRUE - 6e-12, FDOT_TRUE + 6e-12]

LN_F_RANGE = [np.log(F_RANGE[0]), np.log(F_RANGE[1])]
LN_FDOT_RANGE = [np.log(FDOT_RANGE[0]), np.log(FDOT_RANGE[1])]

A_SCALE = A_RANGE[1] - A_RANGE[0]
LN_F_SCALE = LN_F_RANGE[1] - LN_F_RANGE[0]
LN_FDOT_SCALE = LN_FDOT_RANGE[1] - LN_FDOT_RANGE[0]


RANGES = [A_RANGE, LN_F_RANGE, LN_FDOT_RANGE]

PRIOR = PriorDict(
    dict(
        a=Uniform(*A_RANGE),
        ln_f=Uniform(*LN_F_RANGE),
        ln_fdot=Uniform(*LN_FDOT_RANGE),
    )
)

# PRIOR = PriorDict(dict(
#     a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE*10,  minimum=1e-25, maximum=1e-15),
#     ln_f=Gaussian(mu=LN_F_TRUE, sigma=LN_F_SCALE*10),
#     ln_fdot=Gaussian(mu=LN_FDOT_TRUE, sigma=LN_FDOT_SCALE*10)
# ))

CENTERED_PRIOR = PriorDict(
    dict(
        a=TruncatedGaussian(
            mu=A_TRUE, sigma=A_SCALE, minimum=1e-25, maximum=1e-15
        ),
        ln_f=Gaussian(mu=LN_F_TRUE, sigma=LN_F_SCALE),
        ln_fdot=Gaussian(mu=LN_FDOT_TRUE, sigma=LN_FDOT_SCALE),
    )
)


OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)
