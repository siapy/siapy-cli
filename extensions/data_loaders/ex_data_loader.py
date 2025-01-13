import numpy as np
from source.analysis.base import BaseDataLoader


class DataLoaderExample(BaseDataLoader):
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        # For the sake of this example, we will generate random data
        X = np.random.rand(100, 10)
        y = np.random.choice(["label_0", "label_1"], 100)
        return X, y
