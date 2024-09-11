import numpy as np
import pandas as pd
from rich.progress import track
from siapy.transformations.image import rescale

from source.core import settings
from source.helpers import load_reflectance_images, save_spectral_signatures


def create_spectral_signatures(average_pixels: bool):
    image_set_cam1, image_set_cam2 = load_reflectance_images()
    signatures_data = []

    for image_cam1, image_cam2 in track(
        zip(image_set_cam1, image_set_cam2), description="Saving spectral signatures..."
    ):
        filename_cam1 = image_cam1.filepath.stem
        image_idx_cam1, object_idx_cam1 = filename_cam1.split(
            settings.labels_part_deliminator
        )[0].split(settings.labels_between_deliminator)

        filename_cam2 = image_cam2.filepath.stem
        image_idx_cam2, object_idx_cam2 = filename_cam2.split(
            settings.labels_part_deliminator
        )[0].split(settings.labels_between_deliminator)

        assert (
            image_idx_cam1 == image_idx_cam2 and object_idx_cam1 == object_idx_cam2
        ), "Something went wrong. Internal error. Indices should be equal."

        if average_pixels:
            signal_cam1 = image_cam1.mean(axis=(0, 1))
            signal_cam2 = image_cam2.mean(axis=(0, 1))
            filename = filename_cam1.split(settings.labels_part_deliminator)[1]
            label = filename.split(settings.labels_between_deliminator)[
                int(object_idx_cam1) - 1
            ]

            signatures_data.append(
                {
                    "filename": filename,
                    "label": label,
                    "image_idx": image_idx_cam1,
                    "object_idx": object_idx_cam1,
                    "signature": np.concatenate((signal_cam1, signal_cam2)).tolist(),
                }
            )

        else:
            image_np_cam1 = image_cam1.to_numpy()
            image_np_cam2 = image_cam2.to_numpy()
            image_np_cam2 = rescale(image_np_cam2, image_np_cam1.shape[:2])
            filename = filename_cam1.split(settings.labels_part_deliminator)[1]
            label = filename.split(settings.labels_between_deliminator)[
                int(object_idx_cam1) - 1
            ]

            for x, y in np.ndindex(image_np_cam1.shape[:2]):
                signatures_data.append(
                    {
                        "filename": filename,
                        "label": label,
                        "x": x,
                        "y": y,
                        "image_idx": image_idx_cam1,
                        "object_idx": object_idx_cam1,
                        "signature": np.concatenate(
                            (image_np_cam1[x, y, :], image_np_cam2[x, y, :])
                        ).tolist(),
                    }
                )

        save_spectral_signatures(pd.DataFrame(signatures_data))
