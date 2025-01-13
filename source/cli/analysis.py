from functools import reduce
from typing import Optional

import typer
from rich import print
from source.analysis import metrics
from source.analysis.artifacts import artifacts
from source.analysis.extensions import (
    import_data_loader,
    import_model,
    import_parameters,
)
from source.analysis.params import DirParams
from source.analysis.trainer import Trainer

app = typer.Typer()


@app.command()
def test_load_data(
    data_loader: Optional[str] = None,
):
    loader = import_data_loader(data_loader)
    loader.load_data()
    print(
        f"Signatures shape: {loader.load_data()[0].shape},"
        f"\nTargets shape: {loader.load_data()[1].shape}"
        f"\nTargets unique values: {list(set(loader.load_data()[1]))}"
    )


@app.command()
def train_model(
    model: Optional[str] = None,
    data_loader: Optional[str] = None,
    do_optimize: bool = False,
    parameters: Optional[list[str]] = None,
):
    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=do_optimize,
            data_loader_name=data_loader,
        )
    )

    X, y = import_data_loader(data_loader).load_data()
    model_init = import_model(model)
    trainer = Trainer(model_init)

    if do_optimize:
        params = reduce(lambda x, y: x + y, [import_parameters(p) for p in parameters])
        study = trainer.optimize(X, y, params)
        artifacts.save_study(study)
    else:
        score = trainer.score_model(X, y)
        artifacts.save_metric(score)
    artifacts.save_encoder(trainer.encoder)


@app.command()
def generate_metrics(
    model: Optional[str] = None,
    data_loader: Optional[str] = None,
    do_optimize: bool = False,
):
    artifacts.set_dir_params(
        DirParams(
            estimator_name=model,
            estimator_is_optimized=do_optimize,
            data_loader_name=data_loader,
        )
    )

    X, y = import_data_loader(data_loader).load_data()

    model_ = artifacts.load_unfit_model()
    encoder = artifacts.load_encoder()

    metrics_ = metrics.calculate_metrics(model_, encoder, X, y)
    artifacts.save_metrics(metrics_)
