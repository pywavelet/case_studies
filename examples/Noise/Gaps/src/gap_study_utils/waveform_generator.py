from abc import ABC, abstractmethod


class WaveformGenerator(ABC):
    """Class to generate waveforms."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, theta):
        pass


