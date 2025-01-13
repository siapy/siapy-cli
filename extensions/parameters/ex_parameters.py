from siapy.optimizers.parameters import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
)
from source.analysis.base import BaseOptParameters

# Preprocessing


class ParamsSavgol(BaseOptParameters):
    parameters = [
        IntParameter(name="savgol__win_length", low=3, high=15, step=2),
    ]


class ParamsFFT(BaseOptParameters):
    parameters = [
        FloatParameter(name="fft__shape_param", low=0.1, high=2, step=0.1),
        FloatParameter(name="fft__sigma", low=0.1, high=5, step=0.01),
    ]


# Dimensionality reduction


class ParamsPLS(BaseOptParameters):
    parameters = [
        IntParameter(name="pls__n_components", low=3, high=20),
    ]


class ParamsICA(BaseOptParameters):
    parameters = [
        FloatParameter(name="ica__tol", low=1e-4, high=1e-1, log=True),
        IntParameter(name="ica__n_components", low=3, high=20),
        IntParameter(name="ica__max_iter", low=200, high=1000),
        CategoricalParameter(name="ica__algorithm", choices=["parallel", "deflation"]),
        CategoricalParameter(name="ica__fun", choices=["logcosh", "exp", "cube"]),
        CategoricalParameter(name="ica__random_state", choices=[0]),
    ]


# Classifiers


class ParamsXGB(BaseOptParameters):
    parameters = [
        IntParameter(name="xgb__n_estimators", low=100, high=1000),
        IntParameter(name="xgb__max_depth", low=3, high=10),
        FloatParameter(name="xgb__learning_rate", low=1e-3, high=10, log=True),
        FloatParameter(name="xgb__min_child_weight", low=1e-1, high=10.0, log=True),
        FloatParameter(name="xgb__subsample", low=0.5, high=1.0),
        FloatParameter(name="xgb__colsample_bytree", low=0.5, high=1.0),
        FloatParameter(name="xgb__reg_lambda", low=1.0, high=10.0, log=True),
        FloatParameter(name="xgb__gamma", low=1e-3, high=5.0, log=True),
        FloatParameter(name="xgb__reg_alpha", low=1e-3, high=5.0, log=True),
        CategoricalParameter(name="xgb__random_state", choices=[0]),
    ]


class ParamsSVC(BaseOptParameters):
    parameters = [
        FloatParameter(name="svc__C", low=0.1, high=10000, log=True),
        FloatParameter(name="svc__gamma", low=1e-7, high=1, log=True),
        CategoricalParameter(name="svc__random_state", choices=[0]),
    ]
