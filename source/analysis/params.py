from pathlib import Path

from pydantic import BaseModel, model_validator
from source.analysis.extensions import DEFAULT_DATA_LOADER_NAME, DEFAULT_ESTIMATOR_NAME
from typing_extensions import Self


class DirParams(BaseModel):
    estimator_name: str | None = None
    estimator_is_optimized: bool = False
    data_loader_name: str | None = None

    @model_validator(mode="after")
    def set_default_names(self) -> Self:
        if self.estimator_name is None:
            self.estimator_name = DEFAULT_ESTIMATOR_NAME
        if self.data_loader_name is None:
            self.data_loader_name = DEFAULT_DATA_LOADER_NAME
        return self


def params_to_path(params: DirParams) -> str:
    return "__".join(
        [
            params.estimator_name or "",
            params.data_loader_name or "",
            str(params.estimator_is_optimized),
        ]
    )


def params_from_path(path: str | Path) -> DirParams:
    path = Path(path)
    path = path.stem
    parts = path.split("__")

    estimator_name = parts[0]
    data_loader_name = parts[1]
    estimator_is_optimized = parts[2].lower() == "true"

    return DirParams(
        estimator_name=estimator_name,
        estimator_is_optimized=estimator_is_optimized,
        data_loader_name=data_loader_name,
    )
