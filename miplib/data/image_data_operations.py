import logging

import numpy as np
import scipy.ndimage as ndimage

import miplib.processing.itk as itkutils
from miplib.data.containers.image_data_store import (
    ImageDataStore,
    ImageKey,
    ImageType,
)

logger = logging.getLogger(__name__)


def create_rescaled_images(
    store: ImageDataStore,
    image_type: ImageType,
    scale: int,
    chunk_size: tuple[int, ...] | None = None,
) -> None:
    """Downsample every image of *image_type* to *scale* percent of full size.

    Reads the 100%-scale reference, zooms with cubic interpolation,
    adjusts spacing, and writes the result.  Existing datasets at the
    target scale are overwritten.
    """
    existing_scales = store.get_scales(image_type)
    if scale in existing_scales and scale != 100:
        logger.info(
            "Scale %i already exists for image type %s; overwriting.",
            scale,
            image_type,
        )

    z_factor = float(scale) / 100

    n_images = store.get_number_of_images(image_type)
    for index in range(n_images):
        for channel in range(store.channel_count):
            key_new = ImageKey(image_type, index, channel, scale)
            key_ref = ImageKey(image_type, index, channel, 100)

            if store.check_if_exists(key_new):
                store.delete_dataset(key_new)

            data_ref = store.get_image_data(key_ref)
            attrs_ref = store.get_image_attributes(key_ref)

            zoom = (z_factor,) * data_ref.ndim
            data_scaled = ndimage.zoom(data_ref, zoom, order=3)

            new_spacing = tuple(100 * x / scale for x in attrs_ref["spacing"])

            store.add_image(
                key_new,
                data_scaled,
                attrs_ref["angle"],
                new_spacing,
                chunk_size,
            )


def calculate_missing_psfs(store: ImageDataStore) -> None:
    """Synthesize missing PSFs by rotating the first PSF with per-view transforms.

    For each registered view that lacks a PSF, the view 0 PSF is rotated
    using the view's spatial transform and saved with ``calculated=True``.
    """
    max_scale = max(store.get_scales(ImageType.REGISTERED))
    if max_scale < 100:
        logger.warning(
            "No full-resolution registration available; using scale %i.",
            max_scale,
        )

    for channel in range(store.channel_count):
        key_psf_ref = ImageKey(ImageType.PSF, 0, channel, 100)
        psf_data = store.get_image_data(key_psf_ref)
        attrs_psf = store.get_image_attributes(key_psf_ref)
        image_spacing = attrs_psf["spacing"]

        n_registered = store.get_number_of_images(ImageType.REGISTERED)
        for index in range(1, n_registered):
            key_psf_new = ImageKey(ImageType.PSF, index, channel, 100)
            if store.check_if_exists(key_psf_new):
                continue

            key_reg = ImageKey(ImageType.REGISTERED, index, channel, max_scale)
            attrs_reg = store.get_image_attributes(key_reg)

            transform = itkutils.make_itk_transform(
                attrs_reg["tfm_type"],
                psf_data.ndim,
                attrs_reg["tfm_params"],
                attrs_reg["tfm_fixed_params"],
            )

            psf_new = itkutils.rotate_psf(
                psf_data, transform, image_spacing, return_numpy=True
            )

            store.add_image(
                key_psf_new,
                psf_new,
                attrs_reg["angle"],
                image_spacing,
                calculated=True,
            )


def copy_registration_result(
    store: ImageDataStore,
    from_scale: int,
    to_scale: int,
) -> None:
    """Migrate registration transforms from one scale level to another.

    Resamples original images at *to_scale* using the transforms stored
    at *from_scale*, saving the results as registered images.
    """
    if from_scale not in store.get_scales(ImageType.REGISTERED):
        raise ValueError(f"No registration result at scale {from_scale}")

    if to_scale not in store.get_scales(ImageType.ORIGINAL):
        create_rescaled_images(store, ImageType.ORIGINAL, to_scale)

    for channel in range(store.channel_count):
        # View 0 is the reference: copy original to registered (identity).
        key_orig0 = ImageKey(ImageType.ORIGINAL, 0, channel, to_scale)
        data0 = store.get_image_data(key_orig0)
        attrs0 = store.get_image_attributes(key_orig0)

        key_reg0 = ImageKey(ImageType.REGISTERED, 0, channel, to_scale)
        store.add_image(key_reg0, data0, 0, attrs0["spacing"])

        # Get reference ITK image for resampling remaining views.
        key_reg0_ref = ImageKey(ImageType.REGISTERED, 0, channel, to_scale)
        reference = itkutils.convert_from_numpy(
            store.get_image_data(key_reg0_ref),
            store.get_image_attributes(key_reg0_ref)["spacing"],
        )

        n_original = store.get_number_of_images(ImageType.ORIGINAL)
        for view in range(1, n_original):
            key_reg_from = ImageKey(ImageType.REGISTERED, view, channel, from_scale)
            attrs_reg_from = store.get_image_attributes(key_reg_from)

            transform = itkutils.make_itk_transform(
                attrs_reg_from["tfm_type"],
                store.get_image_data(key_reg_from).ndim,
                attrs_reg_from["tfm_params"],
                attrs_reg_from["tfm_fixed_params"],
            )

            key_orig = ImageKey(ImageType.ORIGINAL, view, channel, to_scale)
            itk_image = itkutils.convert_from_numpy(
                store.get_image_data(key_orig),
                store.get_image_attributes(key_orig)["spacing"],
            )

            resampled = itkutils.resample_image(
                itk_image, transform, reference=reference
            )
            result = itkutils.convert_from_itk_image(resampled)

            key_reg_new = ImageKey(ImageType.REGISTERED, view, channel, to_scale)
            store.add_image(
                key_reg_new,
                np.asarray(result),
                attrs_reg_from["angle"],
                store.get_image_attributes(key_orig)["spacing"],
            )

            store.add_transform(
                view,
                channel,
                to_scale,
                attrs_reg_from["tfm_params"],
                attrs_reg_from["tfm_fixed_params"],
                attrs_reg_from["tfm_type"],
            )
