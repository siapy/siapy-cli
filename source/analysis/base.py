from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel
from siapy.optimizers.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    TrialParameters,
)
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline

__ALL__ = [
    "BaseDataLoader",
    "BaseSklearnModel",
    "BaseSklearnPipelineModel",
    "BaseOptParameters",
]


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


class BaseSklearnModel(BaseEstimator, ABC):
    """
    Base class for models.

    This class should be subclassed by models to implement the necessary methods.
    """

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        raise NotImplementedError("Subclasses must implement this method")


class BaseSklearnPipelineModel(BaseSklearnModel):
    """
    Base class for models that are implemented as a pipeline.

    This class should be subclassed by models that are implemented as a pipeline.

    """

    pipeline: Pipeline

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.pipeline = clone(self.pipeline)
        self.pipeline.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.pipeline.score(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.pipeline.fit_transform(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.decision_function(X)

    @property
    def named_steps(self):
        return self.pipeline.named_steps

    @property
    def steps(self):
        return self.pipeline.steps

    def get_params(self, deep: bool = True):
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        if deep:
            # Add parameters of the pipeline steps
            for name, step in self.pipeline.named_steps.items():
                for key, value in step.get_params(deep=True).items():
                    params[f"{name}__{key}"] = value
        return params

    def set_params(self, **params):
        """Set the parameters of this estimator."""
        self.pipeline.set_params(**params)
        return self


class BaseOptParameters(ABC):
    """
    A base class for optimization parameters.

    Attributes:
        parameters (list[BaseModel]): A list of parameters to optimize.
    """

    parameters: list[BaseModel]

    def get_trial_parameters(self) -> TrialParameters:
        trial_parameters = TrialParameters(
            float_parameters=[], int_parameters=[], categorical_parameters=[]
        )

        for p in self.parameters:
            if isinstance(p, FloatParameter):
                trial_parameters.float_parameters.append(p)
            elif isinstance(p, IntParameter):
                trial_parameters.int_parameters.append(p)
            elif isinstance(p, CategoricalParameter):
                trial_parameters.categorical_parameters.append(p)

        return trial_parameters

    def __add__(self, other: "BaseOptParameters") -> "BaseOptParameters":
        combined = BaseOptParameters()
        combined.parameters = self.parameters.copy()
        combined.parameters.extend(other.parameters)
        return combined
