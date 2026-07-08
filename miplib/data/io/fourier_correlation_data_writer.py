import os

import h5py

from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationDataCollection,
)
from miplib.data.containers.image import Image


class FourierCorrelationDataWriter:
    """A class for writing Fourier Correlation Data into an HDF5 file."""

    def __init__(self, output_dir, filename, append=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_path = os.path.join(output_dir, filename)
        if not output_path.endswith(".hdf5"):
            raise ValueError(f"Output path must end with .hdf5, got {output_path}")

        if os.path.isfile(output_path):
            self.data = h5py.File(output_path, mode="r+" if append else "w")
        else:
            self.data = h5py.File(output_path, mode="w")

    def __del__(self):
        self.close()

    def write_metadata(self, metadata):
        """Write a metadata dictionary to the HDF5 file header attributes."""
        if not isinstance(metadata, dict):
            raise TypeError(f"Expected dict, got {type(metadata).__name__}")
        for key, value in metadata.items():
            self.data.attrs[key] = value

    def write_images(self, images):
        """Write Image objects to the data structure."""
        if not isinstance(images, tuple):
            images = (images,)
        for image in images:
            if not isinstance(image, Image):
                raise TypeError(f"Expected Image, got {type(image).__name__}")

        group = self.data.require_group("images")

        for idx, image in enumerate(images):
            image_name = f"image_{idx}"
            group.create_dataset(image_name, data=image)
            spacing_parts = " ".join(f"{s:g}" for s in image.spacing)
            if image.ndim == 2:
                group[image_name].attrs["pixel_size"] = f"{spacing_parts} (yx)"
            else:
                group[image_name].attrs["pixel_size"] = f"{spacing_parts} (zyx)"

    def write_data_set(self, data):
        """Write Fourier Correlation Data to the data structure."""
        if not isinstance(data, FourierCorrelationDataCollection):
            raise TypeError(
                f"Expected FourierCorrelationDataCollection, got {type(data).__name__}"
            )

        group_prefix = "data_set_"

        for angle, data_set in data:
            group_name = group_prefix + angle
            if group_name in self.data:
                raise ValueError(
                    f"Dataset {angle} already exists in the file. "
                    f"Remove it first or use a different name."
                )

            data_set_group = self.data.create_group(group_name)
            resolution_group = data_set_group.create_group("resolution")
            correlation_group = data_set_group.create_group("correlation")

            resolution_group.create_dataset(
                "threshold", data=data_set.resolution["threshold"]
            )
            resolution_group.attrs["resolution"] = data_set.resolution["resolution"]
            point_str = " ".join(
                f"{c:g}" for c in data_set.resolution["resolution-point"]
            )
            resolution_group.attrs["resolution-point"] = f"{point_str} (yx)"
            resolution_group.attrs["criterion"] = data_set.resolution["criterion"]
            resolution_group.create_dataset(
                "resolution-threshold-coefficients",
                data=data_set.resolution["resolution-threshold-coefficients"],
            )

            correlation_group.create_dataset(
                "correlation", data=data_set.correlation["correlation"]
            )
            correlation_group.create_dataset(
                "frequency", data=data_set.correlation["frequency"]
            )
            correlation_group.create_dataset(
                "points-x-bin", data=data_set.correlation["points-x-bin"]
            )
            correlation_group.create_dataset(
                "curve-fit", data=data_set.correlation["curve-fit"]
            )
            correlation_group.create_dataset(
                "curve-fit-coefficients",
                data=data_set.correlation["curve-fit-coefficients"],
            )

    def close(self):
        """Close the HDF5 file."""
        self.data.close()
