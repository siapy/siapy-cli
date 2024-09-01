from siapy.entities import Pixels
from siapy.utils.plots import (
    pixels_select_lasso,
)

from source.core import logger
from source.helpers import (
    extract_labels_from_spectral_images,
    get_images_by_label,
    read_spectral_images,
)


def select_areas_on_images(label: str) -> list[Pixels]:
    image_set_cam1, image_set_cam2 = read_spectral_images()
    labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
        image_set_cam1, image_set_cam2
    )
    image_cam1, _ = get_images_by_label(
        label, image_set_cam1, image_set_cam2, labels_cam1, labels_cam2
    )
    selected_areas = pixels_select_lasso(image_cam1)
    logger.info(f"Selected '{len(selected_areas)}' areas")
    return selected_areas
