import logging
import os

import pims
import SimpleITK as sitk

import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image

logger = logging.getLogger(__name__)


def get_image(
    filename: str,
    series: int = 0,
    channel: int = 0,
    return_type: str = "image",
) -> Image | sitk.Image:
    """Read an image from disk via bioformats (or ITK for .mha files)."""
    if return_type not in ("itk", "image"):
        raise ValueError(
            f"Unsupported return_type {return_type!r}; use 'itk' or 'image'"
        )

    if filename.endswith(".mha"):
        return __itk_image(filename, return_type == "itk")

    return __bioformats(filename, series, channel, return_type == "itk")


def __itk_image(filename: str, return_itk: bool = True) -> sitk.Image | Image:
    """Read an ITK-supported image (.mha/.mhd)."""
    if not filename.endswith((".mha", ".mhd")):
        raise ValueError(f"Expected .mha or .mhd extension, got {filename}")
    image = sitk.ReadImage(filename)
    if return_itk:
        return image
    return itkutils.convert_from_itk_image(image)


def __itk_transform(path: str, return_itk: bool = False) -> tuple | sitk.Transform:
    """Read an ITK spatial transform from disk."""
    if not os.path.isfile(path):
        raise ValueError(f"Not a valid path: {path}")

    transform = sitk.ReadTransform(path)

    if return_itk:
        return transform

    transform_type = transform.GetName()
    params = transform.GetParameters()
    fixed_params = transform.GetFixedParameters()
    return transform_type, params, fixed_params


def __bioformats(
    filename: str,
    series: int = 0,
    channel: int = 0,
    return_itk: bool = False,
) -> Image | sitk.Image:
    """Read an image through the Bioformats importer (most microscopy formats)."""
    if not pims.bioformats.available():
        raise ImportError(
            "jpype is required for the bioformats reader. "
            "Install with: uv sync --group dev"
        )
    reader = pims.bioformats.BioformatsReader(filename, series=series)

    # Bioformats may interpret z-slices as channels if the file lacks OME-XML
    # metadata (common with plain tifffile-written z-stacks). Bundle 'c' as 'z'.
    if "c" in reader.sizes and "z" not in reader.axes:
        reader.bundle_axes = "cyx"

    if "z" not in reader.axes:
        spacing: tuple[float, ...] = (
            reader.metadata.PixelsPhysicalSizeY(0),
            reader.metadata.PixelsPhysicalSizeX(0),
        )
    else:
        spacing = (
            reader.metadata.PixelsPhysicalSizeZ(0),
            reader.metadata.PixelsPhysicalSizeY(0),
            reader.metadata.PixelsPhysicalSizeX(0),
        )

    if "c" in reader.sizes:
        reader.iter_axes = "c"
        if not len(reader) > channel:
            raise IndexError(
                f"Requested channel {channel} but image has only "
                f"{len(reader)} channels."
            )
        reader = reader[channel]
    else:
        reader = reader[0]

    # When OME metadata is missing, bioformats cannot label RGB axes as
    # channels and returns a spatial+colour array (e.g. (3, H, W)) whose
    # ndim exceeds len(spacing).  Collapse to a single spatial image via
    # max projection — a heuristic that preserves signal.
    extra = reader.ndim - len(spacing)
    while reader.ndim > len(spacing):
        if reader.shape[0] in (3, 4):
            reader = reader.max(axis=0)
        elif reader.shape[-1] in (3, 4):
            reader = reader.max(axis=-1)
        else:
            # fallback: collapse leading extra axis
            reader = reader.max(axis=0)
    if extra > 0:
        logger.info(
            "Collapsed %d non-spatial axis(es) via max projection "
            "(OME metadata missing — image may contain RGB channels).",
            extra,
        )

    if return_itk:
        return itkutils.convert_from_numpy(reader, spacing)
    return Image(reader, spacing)
