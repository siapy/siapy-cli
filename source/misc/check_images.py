from pathlib import Path

import numpy as np
from siapy.entities.imagesets import SpectralImageSet

from source.core import logger, settings


def check_duplicate_labels(labels):
    labels_out = []
    for label in labels:
        count = labels.count(label)
        if count > 1:
            labels_out.append((label, count))
    return sorted(set(labels_out))


def check_spectral_images():
    file_paths = [
        Path(file) for file in Path(settings.images_dir).glob("*") if file.is_file()
    ]

    header_paths = [
        path
        for idx, path in enumerate(file_paths)
        if file_paths[idx].suffixes[0] == settings.header_file_suffix
    ]
    image_paths = [
        path
        for idx, path in enumerate(file_paths)
        if file_paths[idx].suffixes[0] == settings.image_file_suffix
    ]

    image_set = SpectralImageSet.from_paths(
        header_paths=header_paths, image_paths=image_paths
    )

    # cameras_id = image_set.cameras_id

    image_set_cam1 = image_set.images_by_camera_id(settings.camera1_id)
    image_set_cam2 = image_set.images_by_camera_id(settings.camera2_id)

    labels_cam1 = [
        image_cam1.filepath.name.split(settings.labels_part_deliminator)[0].split(
            settings.labels_between_deliminator
        )
        for image_cam1 in image_set_cam1
    ]
    labels_cam2 = [
        image_cam2.filepath.name.split(settings.labels_part_deliminator)[0].split(
            settings.labels_between_deliminator
        )
        for image_cam2 in image_set_cam2
    ]

    labels_cam1 = list(np.concatenate(labels_cam1))
    labels_cam2 = list(np.concatenate(labels_cam2))

    labels_unique = sorted(set(labels_cam1))
    labels_duplicated = check_duplicate_labels(labels_cam1)

    msg = ""
    msg += f"   Number of images: {len(image_set_cam1)} (cam1), {len(image_set_cam2)} (cam2)\n"
    msg += (
        f"   Number of labels: {len(labels_cam1)} (cam1), {len(labels_cam2)} (cam2)\n"
    )
    msg += f"   Number of unique labels: {len(labels_unique)} \n"
    msg += f"   Labels: \n{str(labels_unique)} \n"
    msg += f"   Duplicated labels: \n{str(labels_duplicated)} \n"
    logger.info(f"Report: \n{msg}")
