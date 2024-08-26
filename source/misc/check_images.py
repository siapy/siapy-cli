import numpy as np

from source.core import logger
from source.helpers import extract_labels_from_spectral_image, read_spectral_images


def check_duplicate_labels(labels):
    labels_out = []
    for label in labels:
        count = labels.count(label)
        if count > 1:
            labels_out.append((label, count))
    return sorted(set(labels_out))


def check_spectral_images():
    image_set_cam1, image_set_cam2 = read_spectral_images()
    labels_cam1 = [
        extract_labels_from_spectral_image(image) for image in image_set_cam1
    ]
    labels_cam2 = [
        extract_labels_from_spectral_image(image) for image in image_set_cam1
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
