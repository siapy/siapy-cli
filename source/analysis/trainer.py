import numpy as np
from optuna import Study
from siapy.optimizers.configs import (
    CreateStudyConfig,
    OptimizeStudyConfig,
    TabularOptimizerConfig,
)
from siapy.optimizers.optimizers import TabularOptimizer
from siapy.optimizers.scorers import Scorer
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder
from source.analysis.base import BaseOptParameters, BaseSklearnModel

CV = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
STUDY_CONFIG = OptimizeStudyConfig(n_trials=200, n_jobs=-1)
STUDY_CREATE = CreateStudyConfig(direction="maximize")
SCORING = "f1_macro"


class Trainer:
    def __init__(self, model: BaseSklearnModel):
        self._optimizer: TabularOptimizer | None = None
        self._model = model

        self._encoder = LabelEncoder()

    @property
    def encoder(self) -> LabelEncoder:
        return self._encoder

    def optimize(
        self, X: np.ndarray, y: np.ndarray, parameters: BaseOptParameters
    ) -> Study:
        trial_parameters = parameters.get_trial_parameters()
        scorer = Scorer.init_cross_validator_scorer(scoring=SCORING, cv=CV, n_jobs=-1)
        configs = TabularOptimizerConfig(
            trial_parameters=trial_parameters,
            scorer=scorer,
            optimize_study=STUDY_CONFIG,
            create_study=STUDY_CREATE,
        )
        y_encoded = self._encoder.fit_transform(y)
        self._optimizer = TabularOptimizer(
            model=self._model,
            configs=configs,
            X=X,
            y=y_encoded,
        )
        study = self._optimizer.run()
        return study

    def score_model(self, X: np.ndarray, y: np.ndarray) -> float:
        scorer = Scorer.init_cross_validator_scorer(scoring=SCORING, cv=CV, n_jobs=-1)
        y_encoded = self._encoder.fit_transform(y)
        return scorer(model=self._model, X=X, y=y_encoded)
