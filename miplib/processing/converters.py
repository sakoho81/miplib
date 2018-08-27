from math import pi


def degrees_to_radians(angle):
    if angle == 0:
        return 0
    else:
        return angle*pi/180


def radians_to_degrees(angle):
    if angle == 0:
        return 0
    else:
        return angle*180.0/pi