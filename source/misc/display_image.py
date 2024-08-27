from typing import Optional

from matplotlib import pyplot as plt

from source.core import logger, settings
from source.helpers import (
    extract_labels_from_spectral_images,
    get_images_by_label,
    read_spectral_images,
)


def display_spectral_image(label: Optional[str] = None):
    image_set_cam1, image_set_cam2 = read_spectral_images()

    if not image_set_cam1 or not image_set_cam2:
        raise ValueError("Image set is empty. Ensure images are loaded first.")

    if not label:
        image_cam1 = image_set_cam1[0]
        image_cam2 = image_set_cam2[0]
    else:
        image_set_cam1, image_set_cam2 = read_spectral_images()
        labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
            image_set_cam1, image_set_cam2
        )
        image_cam1, image_cam2 = get_images_by_label(
            label, image_set_cam1, image_set_cam2, labels_cam1, labels_cam2
        )

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(image_cam1.to_display())  # noqa
    ax[1].imshow(image_cam2.to_display())  # noqa
    plt.show()
