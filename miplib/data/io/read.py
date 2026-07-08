import logging
import os

import pims
import SimpleITK as sitk
import tifffile

import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image

logger = logging.getLogger(__name__)
scale_c = 1.0e6


def get_image(
    filename: str,
    series: int = 0,
    channel: int = 0,
    return_type: str = "image",
    bioformats: bool = True,
) -> Image | sitk.Image | tuple:
    """Read an image from disk, dispatching on file extension."""
    if return_type not in ("itk", "image"):
        raise ValueError(
            f"Unsupported return_type {return_type!r}; use 'itk' or 'image'"
        )

    if filename.endswith(".mha"):
        return __itk_image(filename, return_type == "itk")

    if bioformats:
        return __bioformats(filename, series, channel, return_type == "itk")
    return __tiff(filename, return_type == "itk")


def __itk_image(filename: str, return_itk: bool = True) -> sitk.Image | Image:
    """Read an ITK-supported image (.mha/.mhd)."""
    if not filename.endswith((".mha", ".mhd")):
        raise ValueError(f"Expected .mha or .mhd extension, got {filename}")
    image = sitk.ReadImage(filename)
    if return_itk:
        return image
    return itkutils.convert_from_itk_image(image)


def __tiff(
    filename: str,
    return_itk: bool = False,
) -> tuple | sitk.Image:
    """Read an ImageJ-style 3D TIFF, extracting voxel spacing from tags."""
    if not filename.endswith((".tif", ".tiff")):
        raise ValueError(f"Expected .tif or .tiff extension, got {filename}")

    tags: dict[str, object] = {}
    with tifffile.TiffFile(filename) as tiff:
        images = tiff.asarray()
        page = tiff.pages[0]
        if hasattr(page, "tags"):
            for tag in list(page.tags.values()):  # type: ignore[union-attr]
                tags[tag.name] = tag.value

    z_spacing = _extract_z_spacing(tags)

    # XResolution/YResolution map to numpy Y/X axes (Z, Y, X order).
    # The writer in write.py performs the inverse mapping.
    spacing = (
        z_spacing,
        scale_c / float(tags["x_resolution"][0]),  # type: ignore[index]
        scale_c / float(tags["y_resolution"][0]),  # type: ignore[index]
    )

    if return_itk:
        return itkutils.convert_from_numpy(images, spacing)
    return images, spacing


def _extract_z_spacing(tags: dict[str, object]) -> float:
    """Extract z-spacing from ImageJ-style image_description tag."""
    if "image_description" not in tags:
        logger.warning("No ImageJ image_description tag; using z-spacing = 1.0")
        return 1.0

    image_descriptor = str(tags["image_description"]).split("\n")
    for line in image_descriptor:
        if "spacing" in line:
            return float(line.split("=")[-1])

    logger.warning("No spacing entry in image_description; using z-spacing = 1.0")
    return 1.0


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

    if return_itk:
        return itkutils.convert_from_numpy(reader, spacing)
    return Image(reader, spacing)
