from siapy.entities import Pixels
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from source.core import logger
from source.helpers import (
    extract_labels_from_spectral_images,
    get_images_by_label,
    read_spectral_images,
)


def train_xgboost_model(
    selected_areas: dict[str, dict[str, list[Pixels]]],
) -> tuple[LabelEncoder,XGBClassifier]:
    image_set_cam1, image_set_cam2 = read_spectral_images()
    labels_cam1, labels_cam2 = extract_labels_from_spectral_images(
        image_set_cam1, image_set_cam2
    )
    X = []
    y = []
    for category, data in selected_areas.items():
        for label, pixels_list in data.items():
            image_cam1, _ = get_images_by_label(
                label, image_set_cam1, image_set_cam2, labels_cam1, labels_cam2
            )
            for pixels in pixels_list:
                siagnatures = image_cam1.to_signatures(pixels)
                signal_mean = siagnatures.signals.mean()
                X.append(signal_mean)
                y.append(category)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    model = XGBClassifier().fit(X, y_encoded)
    logger.info("Model trained successfully.")
    return encoder, model
