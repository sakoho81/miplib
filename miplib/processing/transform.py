import SimpleITK as sitk
import math

def make_translation_transforms_from_xy(xs, ys):
    """
    Makes ITK translation transforms from x,y coordinate pairs.
    :param xs: a list, tuple or other iterable of x coordinates
    :param ys: a list, tuple or other iterable of y coordinates
    :return: a list of transforms
    """
    assert len(xs) == len(ys)
    transforms = []

    for x, y in zip(xs, ys):
        tfm = sitk.TranslationTransform(2)
        tfm.SetParameters((x, y))

        transforms.append(tfm)

    return transforms


def rotate_xy_points_lists(xs, ys, radians):
    """
    Rotate two lists of XY coordinates by an angle on XY plane
    :param xs: the x coordinates
    :param ys: the y coordinates
    :param radians: an angle in radians
    :return: return the xs and ys roated by radians.
    """

    def rotate_origin_only(x, y, radians):
        """Only rotate a point around the origin (0, 0)."""
        xx = x * math.cos(radians) + y * math.sin(radians)
        yy = -x * math.sin(radians) + y * math.cos(radians)

        return xx, yy

    def get_point_pairs(xs, ys, radians):
        for x, y in zip(xs, ys):
            yield rotate_origin_only(x, y, radians)

    points = list(get_point_pairs(xs, ys, radians))

    xs_rot = list(i[0] for i in points)
    ys_rot = list(i[1] for i in points)

    return xs_rot, ys_rot
