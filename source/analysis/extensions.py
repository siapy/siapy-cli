from typing import Optional

from source.analysis.base import BaseDataLoader, BaseOptParameters, BaseSklearnModel
from source.core import settings
from source.utils.utils import import_class

_DATA_LOADERS_DIR = settings.extensions_dir / "data_loaders"
_MODELS_DIR = settings.extensions_dir / "models"
_PARAMETERS_DIR = settings.extensions_dir / "parameters"

DEFAULT_DATA_LOADER_NAME = "DataLoaderExample"
DEFAULT_ESTIMATOR_NAME = "SavgolSVC"
DEFAULT_PARAMETERS_NAME = "ParamsSavgolSVC"


def import_data_loader(data_loader_name: Optional[str]) -> BaseDataLoader:
    return import_class(data_loader_name, _DATA_LOADERS_DIR, DEFAULT_DATA_LOADER_NAME)


def import_model(model_name: Optional[str]) -> BaseSklearnModel:
    return import_class(model_name, _MODELS_DIR, DEFAULT_ESTIMATOR_NAME)


def import_parameters(parameters_name: Optional[str]) -> BaseOptParameters:
    return import_class(parameters_name, _PARAMETERS_DIR, DEFAULT_PARAMETERS_NAME)
