import numpy as np
import shap
from sklearn.base import BaseEstimator, clone
from sklearn.preprocessing import LabelEncoder
from source.analysis.tools import smooth_relevances


def extract_values(
    model: BaseEstimator,
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    model = clone(model)
    y_encoded = np.array(encoder.fit_transform(y))
    model.fit(X, y_encoded)  # type: ignore
    explainer = shap.KernelExplainer(model.predict, X)  # type: ignore
    shap_values = explainer.shap_values(X)
    return shap_values


def reduce_features(
    X: np.ndarray,
    shap_values: np.ndarray,
    band_reduction: int,
) -> np.ndarray:
    relevances = smooth_relevances(shap_values)
    top_features_idx = np.argsort(relevances)[::-1][:band_reduction]
    X_reduced = X[:, top_features_idx]
    return X_reduced
