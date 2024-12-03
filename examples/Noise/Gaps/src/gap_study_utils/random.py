"""
Random number generator utilities.
https://blog.scientific-python.org/numpy/numpy-rng/
"""

from numpy.random import SeedSequence, default_rng


def __getattr__(name):
    """Convenience function to allow access to the default_rng instance
    Can be used as

    >>> from random import rng
    >>> rng.integers(0, 10, size=5)

    """
    if name == "rng":
        return Generator.rng


class Generator:
    rng = default_rng()


def seed(seed):
    Generator.rng = default_rng(seed)


def generate_seeds(nseeds):
    return SeedSequence(Generator.rng.integers(0, 2**63 - 1, size=4)).spawn(
        nseeds
    )
