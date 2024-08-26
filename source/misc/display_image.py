from typing import Optional

from matplotlib import pyplot as plt

from source.core import logger, settings
from source.helpers import extract_labels_from_spectral_images, read_spectral_images


def display_spectral_image(label: Optional[str] = None):
    image_set_cam1, image_set_cam2 = read_spectral_images()

    if not image_set_cam1:
        raise ValueError("Image set is empty")

    if not label:
        image_cam1 = image_set_cam1[0]
        image_cam2 = image_set_cam2[0]
    else:
        image_set_cam1, image_set_cam2 = read_spectral_images()
        labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
            image_set_cam1, image_set_cam2
        )
        index_cam1 = -1
        index_cam2 = -1
        for idx, labels_list in enumerate(labels_cam1):
            if label in labels_list:
                index_cam1 = idx
                break
        for idx, labels_list in enumerate(labels_cam2):
            if label in labels_list:
                index_cam2 = idx
                break

        if index_cam1 == -1:
            raise ValueError(f"Label for '{settings.camera1_id}' was not found.")
        if index_cam2 == -1:
            raise ValueError(f"Label for '{settings.camera2_id}' was not found.")

        image_cam1 = image_set_cam1[index_cam1]
        image_cam2 = image_set_cam2[index_cam2]

    logger.info(f"Displaying images for label: '{label}'")
    logger.info(f"Camera1 image path: '{image_cam1.filepath}'")
    logger.info(f"Camera2 image path: '{image_cam2.filepath}'")

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_cam1.to_display())  # noqa
    ax[1].imshow(image_cam2.to_display())  # noqa
    plt.show()
