from pathlib import Path

from siapy.entities.imagesets import SpectralImage, SpectralImageSet

from source.core import settings


def read_spectral_images() -> tuple[list[SpectralImage], list[SpectralImage]]:
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

    return image_set_cam1, image_set_cam2


def extract_labels_from_spectral_image(image: SpectralImage) -> list[str]:
    return image.filepath.name.split(settings.labels_part_deliminator)[0].split(
        settings.labels_between_deliminator
    )


def extract_labels_from_spectral_images(
    image_set_cam1: list[SpectralImage],
    image_set_cam2: list[SpectralImage],
) -> tuple[list[list[str]], list[list[str]]]:
    labels_cam1 = [
        extract_labels_from_spectral_image(image) for image in image_set_cam1
    ]
    labels_cam2 = [
        extract_labels_from_spectral_image(image) for image in image_set_cam2
    ]

    return labels_cam1, labels_cam2
