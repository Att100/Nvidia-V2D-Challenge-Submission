"""NVIDIA Track-3 input preparation and optimizer adapters."""

from .challenge_run import ChallengeRunAdapter
from .generic_video import GenericVideoPreparer
from .rgb_stereo_gray import RGBStereoGrayPreparer

__all__ = [
    "ChallengeRunAdapter",
    "GenericVideoPreparer",
    "RGBStereoGrayPreparer",
]
