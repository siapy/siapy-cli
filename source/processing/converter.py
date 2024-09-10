import numpy as np
from siapy.entities import SpectralImage
from siapy.utils.images import (
    convert_radiance_image_to_reflectance,
)

from source.core import settings
from source.helpers import load_radiance_images, save_reflectance_image


def calculate_correction_factor_from_panel(
    image: SpectralImage,
    panel_reference_reflectance: float,
) -> np.ndarray:
    # !TODO: need to change this in siapy-lib
    # Also correct type for axis
    # also in convert_radiance_image_to_reflectance set default to None

    panel_radiance_mean = image.mean(axis=(0, 1))
    panel_reflectance_mean = np.full(image.bands, panel_reference_reflectance)
    panel_correction = panel_reflectance_mean / panel_radiance_mean
    return panel_correction


def convert_images_to_reflectance():
    image_set_cam1, image_set_cam2 = load_radiance_images()

    # filename_widx = (
    #     f"{index}{settings.labels_between_deliminator}"
    #     f"{idx}{settings.labels_part_deliminator}"
    #     f"{filename_cam1}.hdr"
    # )
    for image in image_set_cam1:
        filename = image.filepath.stem
        # Here the assumption is made that first object is reference panel
        object_idx = filename.split(settings.labels_part_deliminator)[0].split(
            settings.labels_between_deliminator
        )[1]
        if object_idx == "0":
            panel_correction = calculate_correction_factor_from_panel(
                image=image,
                panel_reference_reflectance=settings.panel_reflectance,
            )
            continue

        filepath = f"{filename}{settings.labels_between_deliminator}reflactance.hdr"
        image_np = convert_radiance_image_to_reflectance(
            image=image, panel_correction=panel_correction, save_path=None
        )
        save_reflectance_image(image_np, filepath, image.metadata)