import SimpleITK as sitk
import tifffile

import miplib.processing.itk as itkutils
from miplib.data.containers.image import Image


def image(path: str, image: Image) -> None:
    """Write an Image to disk, dispatching on file extension."""
    if path.endswith((".tiff", ".tif")):
        __tiff(path, image, image.spacing)
    elif path.endswith((".mha", ".mhd")):
        __itk_image(path, image)
    else:
        raise ValueError(
            f"Unsupported file extension: {path}. Expected .tif, .tiff, .mha, or .mhd."
        )


def __itk_image(path: str, image: Image) -> None:
    """Write an Image in ITK format (.mha/.mhd)."""
    sitk.WriteImage(itkutils.convert_to_itk_image(image), path)


def __tiff(path: str, image: Image, spacing: list[float]) -> None:
    """Write a TIFF (auto-converted to BigTIFF if needed)."""
    if image.ndim >= 3:
        with tifffile.TiffWriter(path, ome=True) as tw:
            metadata: dict[str, object] = {
                "axes": "ZYX",
                "PhysicalSizeZ": spacing[0],
                "PhysicalSizeZUnit": "\u00b5m",
            }
            metadata["PhysicalSizeY"] = spacing[1]
            metadata["PhysicalSizeYUnit"] = "\u00b5m"
            metadata["PhysicalSizeX"] = spacing[2]
            metadata["PhysicalSizeXUnit"] = "\u00b5m"
            tw.write(
                image,
                resolution=(1.0 / spacing[1], 1.0 / spacing[2]),
                metadata=metadata,
            )
    else:
        tifffile.imwrite(
            path,
            image,
            imagej=True,
            resolution=(1.0 / spacing[0], 1.0 / spacing[1]),
        )
