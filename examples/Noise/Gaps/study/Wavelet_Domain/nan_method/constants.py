import numpy as np
import os
from bilby.core.prior import Uniform, TruncatedGaussian, Gaussian, PriorDict

ONE_HOUR = 60 * 60
ONE_DAY = 24 * ONE_HOUR
A_TRUE = 1e-21
LN_F_TRUE = np.log(3e-3)
LN_FDOT_TRUE = np.log(1e-8)
TRUES = [A_TRUE, LN_F_TRUE, LN_FDOT_TRUE]

NF = 64
TMAX = 4 * ONE_DAY
START_GAP = TMAX  * 0.4
END_GAP = START_GAP + 6 * ONE_HOUR
A_RANGE = [9e-22, 1e-20]
LN_F_RANGE = [LN_F_TRUE-0.2, LN_F_TRUE+0.2]
LN_FDOT_RANGE = [LN_FDOT_TRUE-0.2, LN_FDOT_TRUE+0.2]

A_SCALE = A_RANGE[1] - A_RANGE[0]
LN_F_SCALE = LN_F_RANGE[1] - LN_F_RANGE[0]
LN_FDOT_SCALE = LN_FDOT_RANGE[1] - LN_FDOT_RANGE[0]


RANGES = [
    A_RANGE,
    LN_F_RANGE,
    LN_FDOT_RANGE
]

PRIOR = PriorDict(dict(
    a=Uniform(*A_RANGE),
    ln_f=Uniform(*LN_F_RANGE),
    ln_fdot=Uniform(*LN_FDOT_RANGE)
))

# PRIOR = PriorDict(dict(
#     a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE*10,  minimum=1e-25, maximum=1e-15),
#     ln_f=Gaussian(mu=LN_F_TRUE, sigma=LN_F_SCALE*10),
#     ln_fdot=Gaussian(mu=LN_FDOT_TRUE, sigma=LN_FDOT_SCALE*10)
# ))

CENTERED_PRIOR =  PriorDict(dict(
    a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE,  minimum=1e-25, maximum=1e-15),
    ln_f=Gaussian(mu=LN_F_TRUE, sigma=LN_F_SCALE),
    ln_fdot=Gaussian(mu=LN_FDOT_TRUE, sigma=LN_FDOT_SCALE)
))


OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)