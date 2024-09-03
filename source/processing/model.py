import numpy as np
from siapy.entities import Pixels
from siapy.transformations import corregistrator
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from source.core import logger
from source.helpers import (
    extract_labels_from_spectral_images,
    get_images_by_label,
    read_spectral_images,
)


def convert_selected_areas_to_train_data(
    selected_areas: dict[str, dict[str, list[Pixels]]],
    transformation_matx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_set_cam1, image_set_cam2 = read_spectral_images()
    labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
        image_set_cam1, image_set_cam2
    )
    X_cam1 = []
    y_cam1 = []
    X_cam2 = []
    y_cam2 = []
    logger.info("Converting areas to train data format.")
    for category, data in selected_areas.items():
        for label, pixels_list in data.items():
            image_cam1, image_cam2, _, _ = get_images_by_label(
                label, image_set_cam1, image_set_cam2, labels_cam1, labels_cam2
            )
            for pixels_cam1 in pixels_list:
                # Extract data for camera 1
                signatures = image_cam1.to_signatures(pixels_cam1)
                signals = signatures.signals.to_numpy()
                X_cam1.extend(signals)
                y_cam1.extend([category] * signals.shape[0])

                # Extract data for camera 2
                pixels_cam2 = corregistrator.transform(pixels_cam1, transformation_matx)
                signatures = image_cam2.to_signatures(pixels_cam2)
                signals = signatures.signals.to_numpy()
                X_cam2.extend(signals)
                y_cam2.extend([category] * signals.shape[0])
    logger.info("Finished formatting.")

    return np.array(X_cam1), np.array(y_cam1), np.array(X_cam2), np.array(y_cam2)


def train_xgboost_model(
    X: np.ndarray, y: np.ndarray
) -> tuple[LabelEncoder, XGBClassifier]:
    logger.info("Training started ...")
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    model = XGBClassifier().fit(X, y_encoded)
    logger.info("Model trained successfully.")
    return encoder, model
