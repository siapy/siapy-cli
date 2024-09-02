import numpy as np
from siapy.transformations import corregistrator
from siapy.utils.enums import InteractiveButtonsEnum
from siapy.utils.plots import (
    display_multiple_images_with_areas,
    pixels_select_lasso,
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from source.core import logger, settings
from source.helpers import (
    extract_labels_from_spectral_images,
    get_images_by_label,
    read_spectral_images,
)


def _handle_out_flag(out_flag: InteractiveButtonsEnum | None, index: int) -> int:
    if out_flag is InteractiveButtonsEnum.SKIP:
        return index + 1
    elif out_flag is InteractiveButtonsEnum.REPEAT:
        return index
    else:
        return index + 1


def perform_segmentation(
    encoder_cam1: LabelEncoder,
    model_cam1: XGBClassifier,
    encoder_cam2: LabelEncoder,
    model_cam2: XGBClassifier,
    transformation_matx: np.ndarray,
    label: str | None,
):
    image_set_cam1, image_set_cam2 = read_spectral_images()

    if label is None:
        index = 0
    else:
        labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
            image_set_cam1, image_set_cam2
        )
        _, _, index, _ = get_images_by_label(
            label, image_set_cam1, image_set_cam2, labels_cam1, labels_cam2
        )

    while index < len(image_set_cam1) and index < len(image_set_cam2):
        image_cam1 = image_set_cam1[index]
        image_cam2 = image_set_cam2[index]

        logger.info(f"Processed index: '__ {index} __' ")
        logger.info(
            "Processed files:\n" " -> Camera 1 '{}'\n" " -> Camera 2 '{}'".format(
                image_cam1.filepath.stem, image_cam2.filepath.stem
            )
        )

        selected_areas_cam1 = pixels_select_lasso(image_cam1)
        selected_areas_cam2 = [
            corregistrator.transform(pixels_cam1, transformation_matx)
            for pixels_cam1 in selected_areas_cam1
        ]

        out_flag = display_multiple_images_with_areas(
            [
                (image_cam1, selected_areas_cam1),
                (image_cam2, selected_areas_cam2),
            ]
        )
        logger.info(f"Enumerate flag: '{out_flag}'")
        index = _handle_out_flag(out_flag, index)
