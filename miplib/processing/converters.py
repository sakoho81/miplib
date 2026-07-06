from math import pi


def degrees_to_radians(angle: float) -> float:
    """Convert an angle from degrees to radians."""
    if angle == 0:
        return 0
    else:
        return angle * pi / 180


def radians_to_degrees(angle: float) -> float:
    """Convert an angle from radians to degrees."""
    if angle == 0:
        return 0
    else:
        return angle * 180.0 / pi
