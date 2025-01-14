import numpy as np
from sklearn.datasets import make_classification
from source.analysis.base import BaseDataLoader


class DataLoaderExample(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        # Generate a classification dataset with 200 features, 20 informative features,
        # and 100 samples using sklearn
        X, numeric_labels = make_classification(
            n_samples=100,
            n_features=200,
            n_informative=20,
            n_redundant=180,
            random_state=42,
        )
        labels = np.array([f"label_{label}" for label in numeric_labels])
        return X, labels
