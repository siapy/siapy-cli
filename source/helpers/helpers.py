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
