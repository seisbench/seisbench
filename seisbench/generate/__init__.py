"""
This is the docstring for the seisbench.generate module.
"""

from .generator import GenericGenerator
from .augmentation import (
    Normalize,
    Filter,
    FilterKeys,
    FixedWindow,
    WindowAroundSample,
    SlidingWindow,
    RandomWindow,
    ChangeDtype,
    SupervisedLabeller,
    PickLabeller,
)

__all__ = [
    "GenericGenerator",
    "Normalize",
    "Filter",
    "FilterKeys",
    "FixedWindow",
    "WindowAroundSample",
    "SlidingWindow",
    "RandomWindow",
    "ChangeDtype",
    "SupervisedLabeller",
    "PickLabeller",
]
