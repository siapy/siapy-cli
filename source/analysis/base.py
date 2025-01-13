from abc import ABC, abstractmethod

import numpy as np


class BaseDataLoader(ABC):
    """
    Load data for analysis.

    This method should be implemented by subclasses to load the necessary data.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - The first element is an ndarray where columns represent spectral bands and rows represent independent signatures.
            - The second element is an ndarray containing the labels corresponding to these signatures.
    """

    @abstractmethod
    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Subclasses must implement this method")
