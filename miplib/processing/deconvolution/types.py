from __future__ import annotations

from enum import Enum


class FusionMode(Enum):
    SUMMATIVE = "summative"
    MULTIPLICATIVE = "multiplicative"


class FirstEstimate(Enum):
    CONSTANT = "constant"
    IMAGE = "image"
    IMAGE_MEAN = "image_mean"
    AVERAGE = "average"
    SUM = "sum"
