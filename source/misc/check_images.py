from pathlib import Path

import numpy as np
from siapy.entities import SpectralImageSet

from source.core import logger, settings


def parse_labels(filename, labels_path_deliminator, labels_deliminator):
    return filename.split(labels_path_deliminator)[0].split(labels_deliminator)


def check_duplicate_labels(labels):
    labels_out = []
    for label in labels:
        count = labels.count(label)
        if count > 1:
            labels_out.append((label, count))
    return sorted(set(labels_out))


def get_filenames_from_directory(dir_path):
    path = Path(dir_path)
    return [str(file) for file in path.glob("*") if file.is_file()]


def check_spectral_images():
    file_paths = get_filenames_from_directory(settings.images_dir)
    print(file_paths)
    # image_set = SpectralImageSet.from_paths(
    #     header_paths=header_paths, image_paths=image_paths
    # )

    # images_cam1 = data_loader.images.cam1
    # images_cam2 = data_loader.images.cam2

    # filenames_cam1 = [image.filename for image in images_cam1]
    # filenames_cam2 = [image.filename for image in images_cam2]

    # labels_pd = cfg.preparator.labels_path_deliminator
    # labels_d = cfg.preparator.labels_deliminator

    # labels_cam1 = [
    #     parse_labels(filename, labels_pd, labels_d) for filename in filenames_cam1
    # ]
    # labels_cam1 = list(np.concatenate(labels_cam1))
    # labels_cam2 = [
    #     parse_labels(filename, labels_pd, labels_d) for filename in filenames_cam2
    # ]
    # if labels_cam2:
    #     labels_cam2 = list(np.concatenate(labels_cam2))

    # labels_unique = sorted(set(labels_cam1))
    # labels_duplicated = check_duplicate_labels(labels_cam1)

    # msg = ""
    # msg += (
    #     f"   Number of images: {len(images_cam1)} (cam1), {len(images_cam2)} (cam2)\n"
    # )
    # msg += (
    #     f"   Number of labels: {len(labels_cam1)} (cam1), {len(labels_cam2)} (cam2)\n"
    # )
    # msg += f"   Number of unique labels: {len(labels_unique)} \n"
    # msg += f"   Labels: \n{str(labels_unique)} \n"
    # msg += f"   Duplicated labels: \n{str(labels_duplicated)} \n"
    # logger.info(f"Report: \n{msg}")
