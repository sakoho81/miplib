import math
from collections.abc import Iterable

import SimpleITK as sitk


def make_translation_transforms_from_xy(
    xs: Iterable[float], ys: Iterable[float]
) -> list[sitk.Transform]:
    """Return ITK 2D translation transforms for x,y coordinate pairs."""
    xs = list(xs)
    ys = list(ys)
    if len(xs) != len(ys):
        raise ValueError(
            f"Coordinate lists must have equal length, got {len(xs)} vs {len(ys)}"
        )

    transforms: list[sitk.Transform] = []
    for x, y in zip(xs, ys, strict=False):
        tfm = sitk.TranslationTransform(2)
        tfm.SetParameters((x, y))
        transforms.append(tfm)

    return transforms


def rotate_xy_points_lists(
    xs: Iterable[float], ys: Iterable[float], radians: float
) -> tuple[list[float], list[float]]:
    """Rotate XY point lists around the origin by *radians*."""

    def rotate_origin_only(x: float, y: float, a: float) -> tuple[float, float]:
        xx = x * math.cos(a) + y * math.sin(a)
        yy = -x * math.sin(a) + y * math.cos(a)
        return xx, yy

    xs_rot: list[float] = []
    ys_rot: list[float] = []
    for x, y in zip(xs, ys, strict=False):
        rx, ry = rotate_origin_only(x, y, radians)
        xs_rot.append(rx)
        ys_rot.append(ry)

    return xs_rot, ys_rot
