from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from source.utils.utils import dict_zip

METRIC_AVERAGE = "macro"
METRIC_FUNC = {
    "accuracy": accuracy_score,
    "balanced_accuracy_score": balanced_accuracy_score,
    "cohen_kappa_score": cohen_kappa_score,
    "f1": partial(f1_score, average=METRIC_AVERAGE),
    "precision": partial(precision_score, average=METRIC_AVERAGE),
    "recall": partial(recall_score, average=METRIC_AVERAGE),
}
ALL = 1000


@dataclass
class MetricsContainer:
    def __init__(self):
        self._metrics = {key: [] for key in METRIC_FUNC.keys()}

    def calculate(self, y_test: ArrayLike, y_pred: ArrayLike):
        for metric_name, metric_func in METRIC_FUNC.items():
            self._metrics[metric_name].append(metric_func(y_test, y_pred))

    def mean(self) -> dict[str, np.number]:
        return {key: np.mean(item) for key, item in self._metrics.items()}

    def std(self) -> dict[str, np.number]:
        return {key: np.std(item) for key, item in self._metrics.items()}


def cross_validate(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
) -> dict[int, MetricsContainer]:
    meta_unique = np.unique(meta)
    metrics_cont = {key: MetricsContainer() for key in np.append(meta_unique, ALL)}

    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)
    for train_index, test_index in rskf.split(X, y, groups=None):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = clone(model)
        model.fit(x_train, y_train)  # type: ignore
        y_pred = model.predict(x_test)  # type: ignore

        for idx_m in meta_unique:
            metrics_cont[idx_m].calculate(
                y_test[meta[test_index] == idx_m],
                y_pred[meta[test_index] == idx_m],
            )
        metrics_cont[ALL].calculate(y_test, y_pred)

    return metrics_cont


class Metrics(BaseModel):
    name: str
    mean: float
    std: float | None = None
    meta_id: Any | None = None


def calculate_metrics(
    model: BaseEstimator,
    encoder: LabelEncoder,
    X: np.ndarray,
    y: np.ndarray,
    meta: np.ndarray,
) -> list[Metrics]:
    metrics = []
    y_encoded = np.array(encoder.fit_transform(y))
    metrics_cont = cross_validate(model, X, y_encoded, meta)

    for id_, metrics_temp in metrics_cont.items():
        for name_key, mean_val, std_val in dict_zip(
            metrics_temp.mean(), metrics_temp.std()
        ):
            metrics.append(
                Metrics(name=name_key, mean=mean_val, std=std_val, meta_id=id_)
            )

    return metrics
