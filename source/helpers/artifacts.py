import os
import pickle
import shutil
from typing import Any

import numpy as np
import pandas as pd
from siapy.entities import Pixels
from siapy.entities.imagesets import SpectralImage
from siapy.utils.images import save_image
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from source.core import logger, settings

from .helpers import read_spectral_images

_TRANSFORMATION_MATX_FILENAME = settings.artifacts_dir / "transform/matx.pkl"
_SELECTED_AREAS_DIR = settings.artifacts_dir / "areas"
_MODEL_DIR = settings.artifacts_dir / "model"
_MODEL_CLF_FILENAMEM_CAM1 = _MODEL_DIR / "model_cam1.json"
_MODEL_ENCODER_FILENAME_CAM1 = _MODEL_DIR / "encoder_cam1.pkl"
_MODEL_CLF_FILENAMEM_CAM2 = _MODEL_DIR / "model_cam2.json"
_MODEL_ENCODER_FILENAME_CAM2 = _MODEL_DIR / "encoder_cam2.pkl"
_IMAGE_DIR = settings.artifacts_dir / "images"
_IMAGE_RADIANCE_DIR = _IMAGE_DIR / "radiance"
_IMAGE_REFLECTANCE_DIR = _IMAGE_DIR / "reflectance"
_SIGNATURES_EXPORT_DIR = settings.artifacts_dir / "signatures"
_SIGNATURES_REFLECTANCE = _SIGNATURES_EXPORT_DIR / "signatures.parquet"


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


def save_model(
    encoder_cam1: LabelEncoder,
    model_cam1: XGBClassifier,
    encoder_cam2: LabelEncoder,
    model_cam2: XGBClassifier,
):
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_cam1.save_model(_MODEL_CLF_FILENAMEM_CAM1)
    with open(_MODEL_ENCODER_FILENAME_CAM1, "wb") as f:
        pickle.dump(encoder_cam1, f)
    logger.info("Model saved for camera 1.")

    model_cam2.save_model(_MODEL_CLF_FILENAMEM_CAM2)
    with open(_MODEL_ENCODER_FILENAME_CAM2, "wb") as f:
        pickle.dump(encoder_cam2, f)
    logger.info("Model saved for camera 2.")

    # load_model()


def load_model() -> tuple[LabelEncoder, XGBClassifier, LabelEncoder, XGBClassifier]:
    if (
        not _MODEL_CLF_FILENAMEM_CAM1.exists()
        or not _MODEL_ENCODER_FILENAME_CAM1.exists()
    ):
        raise FileNotFoundError("Model files not found for camera 1.")

    model_cam1 = XGBClassifier()
    model_cam1.load_model(_MODEL_CLF_FILENAMEM_CAM1)

    with open(_MODEL_ENCODER_FILENAME_CAM1, "rb") as f:
        encoder_cam1 = pickle.load(f)

    logger.info("Model loaded for camera 1.")

    if (
        not _MODEL_CLF_FILENAMEM_CAM2.exists()
        or not _MODEL_ENCODER_FILENAME_CAM2.exists()
    ):
        raise FileNotFoundError("Model files not found for camera 2.")

    model_cam2 = XGBClassifier()
    model_cam2.load_model(_MODEL_CLF_FILENAMEM_CAM2)

    with open(_MODEL_ENCODER_FILENAME_CAM2, "rb") as f:
        encoder_cam2 = pickle.load(f)

    logger.info("Model loaded for camera 2.")

    return encoder_cam1, model_cam1, encoder_cam2, model_cam2


def save_radiance_image(
    image: np.ndarray, filename: str, metadata: dict[str, Any] | None = None
):
    _IMAGE_RADIANCE_DIR.mkdir(parents=True, exist_ok=True)
    save_image(image, _IMAGE_RADIANCE_DIR / filename, metadata=metadata)


def save_reflectance_image(
    image: np.ndarray, filename: str, metadata: dict[str, Any] | None = None
):
    _IMAGE_REFLECTANCE_DIR.mkdir(parents=True, exist_ok=True)
    save_image(image, _IMAGE_REFLECTANCE_DIR / filename, metadata=metadata)


def load_radiance_images() -> tuple[list[SpectralImage], list[SpectralImage]]:
    if not _IMAGE_RADIANCE_DIR.exists():
        raise FileNotFoundError(
            "Radiance images directory does not exist. You need to segment images first."
        )
    return read_spectral_images(_IMAGE_RADIANCE_DIR)


def load_reflectance_images() -> tuple[list[SpectralImage], list[SpectralImage]]:
    if not _IMAGE_REFLECTANCE_DIR.exists():
        raise FileNotFoundError(
            "Reflectance images directory does not exist. You need to segment images first."
        )
    return read_spectral_images(_IMAGE_REFLECTANCE_DIR)


def save_spectral_signatures(signatures: pd.DataFrame):
    _SIGNATURES_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    signatures.to_parquet(_SIGNATURES_REFLECTANCE, engine="pyarrow")


def load_spectral_signatures() -> pd.DataFrame:
    if not _SIGNATURES_REFLECTANCE.exists():
        raise FileNotFoundError("Spectral signatures file does not exist.")

    return pd.read_parquet(_SIGNATURES_REFLECTANCE, engine="pyarrow")
