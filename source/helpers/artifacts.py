import pickle

import numpy as np

from source.core import logger, settings

_TRANSFORMATION_MATX_FILENAME = settings.artifacts_dir / "transform/matx.pkl"


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
