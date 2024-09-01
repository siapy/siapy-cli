import os
import pickle
import shutil

import numpy as np
from siapy.entities import Pixels
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from source.core import logger, settings

_TRANSFORMATION_MATX_FILENAME = settings.artifacts_dir / "transform/matx.pkl"
_SELECTED_AREAS_DIR = settings.artifacts_dir / "areas"
_MODEL_DIR = settings.artifacts_dir / "model"
_MODEL_CLF_FILENAME = _MODEL_DIR / "model.json"
_MODEL_ENCODER_FILENAME = _MODEL_DIR / "encoder.pkl"


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
        area.save_to_parquet(category_areas_dir / f"{idx}.parquet")


def load_selected_areas(category: str) -> dict[str, list[Pixels]]:
    category_areas_dir = _SELECTED_AREAS_DIR / category
    if not category_areas_dir.exists():
        raise FileNotFoundError(
            f"Selected areas directory not found: '{category_areas_dir}'"
        )
    logger.info(f"Loading selected areas from: '{category_areas_dir}'")

    selected_areas = {}
    labels = category_areas_dir.glob("*")
    for label in labels:
        category_areas_dir_ = category_areas_dir / label.stem
        area_dfs = []
        for area_file in sorted(category_areas_dir_.glob("*.parquet")):
            area_df = Pixels.load_from_parquet(area_file)
            area_dfs.append(area_df)
        selected_areas[label.stem] = area_dfs
    return selected_areas


def load_all_selected_areas() -> dict[str, dict[str, list[Pixels]]]:
    categories = _SELECTED_AREAS_DIR.glob("*")
    return {
        category.stem: load_selected_areas(category.stem) for category in categories
    }


def save_model(encoder: LabelEncoder, model: XGBClassifier):
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(_MODEL_CLF_FILENAME)
    with open(_MODEL_ENCODER_FILENAME, "wb") as f:
        pickle.dump(encoder, f)
    logger.info("Model saved.")

    # load_model()


def load_model() -> tuple[LabelEncoder, XGBClassifier]:
    if not _MODEL_CLF_FILENAME.exists() or not _MODEL_ENCODER_FILENAME.exists():
        raise FileNotFoundError("Model files not found.")

    model = XGBClassifier()
    model.load_model(_MODEL_CLF_FILENAME)

    with open(_MODEL_ENCODER_FILENAME, "rb") as f:
        encoder = pickle.load(f)

    logger.info("Model loaded.")
    return encoder, model
