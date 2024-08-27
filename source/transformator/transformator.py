import numpy as np
from siapy.entities.imagesets import SpectralImage
from siapy.transformations import corregistrator
from siapy.utils.plots import pixels_select_click

from source.core import logger
from source.helpers import (
    extract_labels_from_spectral_images,
    get_images_by_label,
    read_spectral_images,
)


def find_transformation_between_images(label: str) -> np.ndarray:
    image_set_cam1, image_set_cam2 = read_spectral_images()

    if not image_set_cam1 or not image_set_cam2:
        raise ValueError("Image set is empty. Ensure images are loaded first.")

    image_set_cam1, image_set_cam2 = read_spectral_images()
    labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
        image_set_cam1, image_set_cam2
    )
    image_cam1, image_cam2 = get_images_by_label(
        label, image_set_cam1, image_set_cam2, labels_cam1, labels_cam2
    )

    pixels_cam1 = pixels_select_click(image_cam1)
    pixels_cam2 = pixels_select_click(image_cam2)

    matx, _ = corregistrator.align(pixels_cam2, pixels_cam1, plot_progress=True)
    logger.info("Transformation matrix calculated")
    # pixels_transformed = corregistrator.transform(pixels_cam1, matx)
    return matx
