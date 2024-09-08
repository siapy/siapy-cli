import time

import numpy as np
from rich.progress import track
from siapy.entities import Pixels, SpectralImage
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
    save_spectral_image,
)


def _handle_out_flag(out_flag: InteractiveButtonsEnum | None, index: int) -> int:
    if out_flag is InteractiveButtonsEnum.SKIP:
        return index + 1
    elif out_flag is InteractiveButtonsEnum.REPEAT:
        return index
    else:
        return index + 1


def make_predictions(
    encoder_cam1: LabelEncoder,
    model_cam1: XGBClassifier,
    encoder_cam2: LabelEncoder,
    model_cam2: XGBClassifier,
    image_cam1: SpectralImage,
    image_cam2: SpectralImage,
    selected_areas_cam1: list[Pixels],
    selected_areas_cam2: list[Pixels],
) -> tuple[list[Pixels], list[Pixels]]:
    # Iterate from 1: to exclude reference panel from segmenting
    selected_areas_cam1_out = [selected_areas_cam1[0]]
    selected_areas_cam2_out = [selected_areas_cam2[0]]

    logger.info("Segmentation started ...")
    # Segment for camera 1
    start_time_cam1 = time.time()
    for idx, area in enumerate(
        track(selected_areas_cam1[1:], description="Processing Camera 1 Areas...")
    ):
        signals = image_cam1.to_signatures(area).signals.to_numpy()
        y_pred = encoder_cam1.inverse_transform(model_cam1.predict(signals))
        mask = y_pred == settings.classification_category_keep
        area_filtered = area.df.iloc[mask].reset_index(drop=True)
        selected_areas_cam1_out.append(Pixels.from_iterable(area_filtered))
    end_time_cam1 = time.time()
    logger.info(
        f"Time taken for Camera 1 segmentation: {end_time_cam1 - start_time_cam1:.2f} seconds"
    )

    # Segment for camera 2
    start_time_cam2 = time.time()
    for idx, area in enumerate(
        track(selected_areas_cam2[1:], description="Processing Camera 2 Areas...")
    ):
        signals = image_cam2.to_signatures(area).signals.to_numpy()
        y_pred = encoder_cam2.inverse_transform(model_cam2.predict(signals))
        mask = y_pred == settings.classification_category_keep
        area_filtered = area.df.iloc[mask].reset_index(drop=True)
        selected_areas_cam2_out.append(Pixels.from_iterable(area_filtered))
    end_time_cam2 = time.time()
    logger.info(
        f"Time taken for Camera 2 segmentation: {end_time_cam2 - start_time_cam2:.2f} seconds"
    )
    logger.info(
        f"Total segmentation time: {end_time_cam2 - start_time_cam1:.2f} seconds"
    )

    return selected_areas_cam1_out, selected_areas_cam2_out


def save_segmented_images(
    index: int,
    image_cam1: SpectralImage,
    image_cam2: SpectralImage,
    selected_areas_cam1: list[Pixels],
    selected_areas_cam2: list[Pixels],
):
    filename_cam1 = image_cam1.filepath.stem
    filename_cam2 = image_cam2.filepath.stem
    meta_keys_to_extract = [
        "description",
        "data type",
        "data ignore value",
        "interleave",
        "default bands",
        "byte order",
        "wavelength",
    ]

    logger.info("Saving segmented images for Camera 1 ...")
    metadata_cam1 = {key: image_cam1.metadata[key] for key in meta_keys_to_extract}
    for idx, area in enumerate(selected_areas_cam1):
        filename_widx = (
            f"{index}{settings.labels_between_deliminator}"
            f"{idx}{settings.labels_part_deliminator}"
            f"{filename_cam1}.hdr"
        )
        subarray = image_cam1.to_subarray(area)
        save_spectral_image(subarray, filename_widx, metadata_cam1)

    logger.info("Saving segmented images for Camera 2 ...")
    metadata_cam2 = {key: image_cam2.metadata[key] for key in meta_keys_to_extract}
    for idx, area in enumerate(selected_areas_cam2):
        filename_widx = (
            f"{index}{settings.labels_between_deliminator}"
            f"{idx}{settings.labels_part_deliminator}"
            f"{filename_cam2}.hdr"
        )
        subarray = image_cam2.to_subarray(area)
        save_spectral_image(subarray, filename_widx, metadata_cam2)


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
        if not selected_areas_cam1 or not selected_areas_cam2:
            logger.warning("No areas were selected. Repeating ...")
            continue

        selected_areas_cam1, selected_areas_cam2 = make_predictions(
            encoder_cam1,
            model_cam1,
            encoder_cam2,
            model_cam2,
            image_cam1,
            image_cam2,
            selected_areas_cam1,
            selected_areas_cam2,
        )

        out_flag = display_multiple_images_with_areas(
            [
                (image_cam1, selected_areas_cam1),
                (image_cam2, selected_areas_cam2),
            ]
        )

        if out_flag is InteractiveButtonsEnum.SAVE:
            logger.info("Saving images ...")
            save_segmented_images(
                index,
                image_cam1,
                image_cam2,
                selected_areas_cam1,
                selected_areas_cam2,
            )

        index = _handle_out_flag(out_flag, index)
