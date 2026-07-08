import os

import h5py

from miplib.data.containers.fourier_correlation_data import (
    FourierCorrelationData,
    FourierCorrelationDataCollection,
)
from miplib.data.containers.image import Image


class FourierCorrelationDataReader:
    """Read Fourier Correlation Data from an HDF5 file."""

    def __init__(self, file_path: str) -> None:
        if not os.path.isfile(file_path) or not file_path.endswith(".hdf5"):
            raise ValueError(f"Not a valid filename: {file_path}")

        self.data = h5py.File(file_path, mode="r")

    def __del__(self) -> None:
        self.close()

    def read_metadata(self) -> dict[str, object]:
        """Read a metadata dictionary from the HDF5 file header."""
        return dict(self.data.attrs)

    def read_images(self) -> list[Image]:
        """Read images from the data structure."""
        if "images" not in self.data:
            raise ValueError("No images to read")

        images: list[Image] = []
        for name in self.data["images"]:
            ds = self.data["images"][name]
            spacing = [
                float(s) for s in ds.attrs["pixel_size"].split()[0 : len(ds.shape)]
            ]
            images.append(Image(ds[:], spacing))

        return images

    def read_data_set(self) -> FourierCorrelationDataCollection:
        """Read FRC/FSC data into a FourierCorrelationDataCollection."""
        group_prefix = "data_set_"
        data_sets = FourierCorrelationDataCollection()
        for group_name in list(self.data.keys()):
            if group_prefix in group_name:
                angle = group_name.split("_")[-1]
                data_set = FourierCorrelationData()
                resolution_group = self.data[group_name]["resolution"]
                correlation_group = self.data[group_name]["correlation"]

                data_set.resolution["threshold"] = resolution_group["threshold"][:]
                data_set.resolution["resolution"] = resolution_group.attrs["resolution"]
                data_set.resolution["resolution-point"] = resolution_group.attrs[
                    "resolution-point"
                ].split()[:-1]
                data_set.resolution["criterion"] = resolution_group.attrs["criterion"]
                data_set.resolution["resolution-threshold-coefficients"] = (
                    resolution_group["resolution-threshold-coefficients"][:]
                )

                data_set.correlation["correlation"] = correlation_group["correlation"][
                    :
                ]
                data_set.correlation["frequency"] = correlation_group["frequency"][:]
                data_set.correlation["points-x-bin"] = correlation_group[
                    "points-x-bin"
                ][:]
                data_set.correlation["curve-fit"] = correlation_group["curve-fit"][:]
                data_set.correlation["curve-fit-coefficients"] = correlation_group[
                    "curve-fit-coefficients"
                ][:]

                data_sets[int(angle)] = data_set

        return data_sets

    def close(self) -> None:
        """Close the HDF5 file."""
        self.data.close()
