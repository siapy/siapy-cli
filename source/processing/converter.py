import numpy as np
from siapy.entities import SpectralImage
from siapy.utils.images import (
    calculate_correction_factor_from_panel,
    convert_radiance_image_to_reflectance,
)

from source.core import settings
from source.helpers import load_radiance_images, save_reflectance_image


def _convert_imageset_to_reflectance(
    image_set: list[SpectralImage], panel_reflectance: float
):
    panel_correction: None | np.ndarray = None
    panel_image_idx: None | str = None

    for image in image_set:
        filename = image.filepath.stem
        image_idx, object_idx = filename.split(settings.labels_part_deliminator)[
            0
        ].split(settings.labels_between_deliminator)
        # Here the assumption is made that first object is reference panel
        if object_idx == "0":
            panel_correction = calculate_correction_factor_from_panel(
                image=image,
                panel_reference_reflectance=panel_reflectance,
            )
            panel_image_idx = image_idx
            continue

        if (
            panel_correction is not None
            and panel_image_idx is not None
            and image_idx == panel_image_idx
        ):
            filepath = f"{filename}{settings.labels_between_deliminator}reflactance.hdr"
            image_np = convert_radiance_image_to_reflectance(
                image=image, panel_correction=panel_correction, save_path=None
            )
            if not isinstance(image_np, np.ndarray):
                raise ValueError(
                    "convert_radiance_image_to_reflectance must return np.ndarray"
                )
            save_reflectance_image(image_np, filepath, image.metadata)
        else:
            raise ValueError(
                "Something went wrong. Internal error. "
                "If reference panel has idx=0 and the paths are sorted correctly this should not happen."
            )


def convert_images_to_reflectance(panel_reflectance: float):
    image_set_cam1, image_set_cam2 = load_radiance_images()
    _convert_imageset_to_reflectance(image_set_cam1, panel_reflectance)
    _convert_imageset_to_reflectance(image_set_cam2, panel_reflectance)
