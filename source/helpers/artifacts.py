import os
import pickle
import shutil

import numpy as np
from siapy.entities import Pixels

from source.core import logger, settings

_TRANSFORMATION_MATX_FILENAME = settings.artifacts_dir / "transform/matx.pkl"
_SELECTED_AREAS_DIR = settings.artifacts_dir / "areas"


def save_transformation_matrix(matx: np.ndarray):
    _TRANSFORMATION_MATX_FILENAME.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving transformation matrix to: %s" % _TRANSFORMATION_MATX_FILENAME)
    with open(_TRANSFORMATION_MATX_FILENAME, "wb") as f:
        pickle.dump(matx, f)


def load_transformation_matrix() -> np.ndarray:
    if not _TRANSFORMATION_MATX_FILENAME.exists():
        raise FileNotFoundError(
            f"Transformation matrix file not found: '{_TRANSFORMATION_MATX_FILENAME}'"
        )
    logger.info(
        "Loading transformation matrix from: %s" % _TRANSFORMATION_MATX_FILENAME
    )
    with open(_TRANSFORMATION_MATX_FILENAME, "rb") as f:
        matx = pickle.load(f)

    return matx


def save_selected_areas(selected_areas: list[Pixels], category: str, label: str):
    category_areas_dir = _SELECTED_AREAS_DIR / category / label
    if os.path.exists(category_areas_dir):
        shutil.rmtree(category_areas_dir)
    category_areas_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving selected areas to: %s" % category_areas_dir)
    for idx, area in enumerate(selected_areas):
        with open(category_areas_dir / f"{idx}.pkl", "wb") as f:
            pickle.dump(area.df, f)


def load_selected_areas(category: str) -> list[Pixels]:
    category_areas_dir = _SELECTED_AREAS_DIR / category
    if not category_areas_dir.exists():
        raise FileNotFoundError(
            f"Selected areas directory not found: '{category_areas_dir}'"
        )
    logger.info(f"Loading selected areas from: '{category_areas_dir}'")

    selected_areas = []
    labels = category_areas_dir.glob("*")
    for label in labels:
        category_areas_dir_ = category_areas_dir / label.stem
        for area_file in sorted(category_areas_dir_.glob("*.pkl")):
            with open(area_file, "rb") as f:
                area_df = pickle.load(f)
                selected_areas.append(Pixels(area_df))
    return selected_areas
