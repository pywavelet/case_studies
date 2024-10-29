import numpy as np
import os
from bilby.core.prior import Uniform, TruncatedGaussian, Gaussian, PriorDict, LogUniform

ONE_HOUR = 60 * 60
ONE_DAY = 24 * ONE_HOUR
A_TRUE = 1e-21
F_TRUE = 3e-3
FDOT_TRUE = 1e-8
TRUES = [A_TRUE, F_TRUE, FDOT_TRUE]

NF = 64
TMAX = 4 * ONE_DAY
START_GAP = TMAX  * 0.4
END_GAP = START_GAP + 6 * ONE_HOUR
A_RANGE = np.sort([A_TRUE*0.5, A_TRUE*1.5])
F_RANGE = np.sort([F_TRUE*0.5, F_TRUE*1.5])
FDOT_RANGE = np.sort([FDOT_TRUE*0.5, FDOT_TRUE*1.5])

A_SCALE = A_RANGE[1] - A_RANGE[0]
F_SCALE = F_RANGE[1] - F_RANGE[0]
FDOT_SCALE = FDOT_RANGE[1] - FDOT_RANGE[0]


RANGES = [
    A_RANGE,
    F_RANGE,
    FDOT_RANGE
]

PRIOR = PriorDict(dict(
    a=LogUniform(*A_RANGE),
    f=LogUniform(*F_RANGE),
    fdot=LogUniform(*FDOT_RANGE)
))

# PRIOR = PriorDict(dict(
#     a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE*10,  minimum=1e-25, maximum=1e-15),
#     f=Gaussian(mu=F_TRUE, sigma=F_SCALE*10),
#     fdot=Gaussian(mu=FDOT_TRUE, sigma=FDOT_SCALE*10)
# ))

CENTERED_PRIOR =  PriorDict(dict(
    a=TruncatedGaussian(mu=A_TRUE, sigma=A_SCALE * 0.01,  minimum=1e-25, maximum=1e-15),
    f=Gaussian(mu=F_TRUE, sigma=F_SCALE* 0.01),
    fdot=Gaussian(mu=FDOT_TRUE, sigma=FDOT_SCALE* 0.01)
))


OUTDIR = "plots"
os.makedirs(OUTDIR, exist_ok=True)