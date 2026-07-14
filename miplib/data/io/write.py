from pathlib import Path

import SimpleITK as sitk
import tifffile

import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image


def image(path: str | Path, image: Image) -> None:
    """Write an Image to disk, dispatching on file extension."""
    p = Path(path)
    if p.suffix in (".tiff", ".tif"):
        __tiff(p, image, image.spacing)
    elif p.suffix in (".mha", ".mhd"):
        __itk_image(p, image)
    else:
        raise ValueError(
            f"Unsupported file extension: {path}. Expected .tif, .tiff, .mha, or .mhd."
        )


def __itk_image(path: str | Path, image: Image) -> None:
    """Write an Image in ITK format (.mha/.mhd)."""
    sitk.WriteImage(itkutils.convert_to_itk_image(image), str(path))


def __tiff(path: str | Path, image: Image, spacing: list[float]) -> None:
    """Write a TIFF with OME metadata (auto-converted to BigTIFF if needed)."""
    if image.ndim == 2:
        axes = "YX"
        resolution: tuple[float, float] = (1.0 / spacing[0], 1.0 / spacing[1])
        metadata: dict[str, object] = {
            "axes": axes,
            "PhysicalSizeY": spacing[0],
            "PhysicalSizeYUnit": "\u00b5m",
            "PhysicalSizeX": spacing[1],
            "PhysicalSizeXUnit": "\u00b5m",
        }
    else:
        axes = "ZYX"
        resolution = (1.0 / spacing[1], 1.0 / spacing[2])
        metadata = {
            "axes": axes,
            "PhysicalSizeZ": spacing[0],
            "PhysicalSizeZUnit": "\u00b5m",
            "PhysicalSizeY": spacing[1],
            "PhysicalSizeYUnit": "\u00b5m",
            "PhysicalSizeX": spacing[2],
            "PhysicalSizeXUnit": "\u00b5m",
        }

    with tifffile.TiffWriter(path, ome=True) as tw:
        tw.write(image, resolution=resolution, metadata=metadata)
