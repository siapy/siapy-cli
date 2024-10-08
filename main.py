import json
from importlib.metadata import PackageNotFoundError, version
from typing import Optional

import typer
from pydantic.json import pydantic_encoder
from source.core import logger, settings
from source.helpers import (
    load_all_selected_areas,
    load_model,
    load_transformation_matrix,
    save_model,
    save_selected_areas,
    save_transformation_matrix,
)
from source.misc import (
    check_spectral_images,
    display_spectral_image,
    display_spectral_images_with_areas,
)
from source.processing import (
    convert_images_to_reflectance,
    convert_selected_areas_to_train_data,
    create_spectral_signatures,
    find_transformation_between_images,
    perform_segmentation,
    select_areas_on_images,
    train_xgboost_model,
)

app = typer.Typer()


def version_callback(value: bool):
    if value:
        try:
            siapy_version = version("siapy")
        except PackageNotFoundError:
            siapy_version = "unknown"
        typer.echo(f"CLI version: 0.1.0\nSiapy library version: {siapy_version}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        callback=version_callback,
        is_eager=True,
        help="Show the application's version and exit.",
    ),
):
    pass


@app.command()
def display_settings():
    logger.info(json.dumps(settings.model_dump(), default=pydantic_encoder, indent=4))


@app.command()
def check_images():
    check_spectral_images()


@app.command()
def display_image(label: Optional[str] = None):
    display_spectral_image(label)


@app.command()
def calculate_transformation(label: str):
    matx = find_transformation_between_images(label)
    save_transformation_matrix(matx)


@app.command()
def select_areas(label: str, category: str):
    selected_areas = select_areas_on_images(label)
    matx = load_transformation_matrix()
    display_spectral_images_with_areas(label, selected_areas, matx)
    save_selected_areas(selected_areas, category, label)


@app.command()
def train_model():
    selected_areas = load_all_selected_areas()
    matx = load_transformation_matrix()
    X_cam1, y_cam1, X_cam2, y_cam2 = convert_selected_areas_to_train_data(
        selected_areas, matx
    )
    encoder_cam1, model_cam1 = train_xgboost_model(X_cam1, y_cam1)
    encoder_cam2, model_cam2 = train_xgboost_model(X_cam2, y_cam2)
    save_model(encoder_cam1, model_cam1, encoder_cam2, model_cam2)


@app.command()
def segment_images(label: Optional[str] = None):
    encoder_cam1, model_cam1, encoder_cam2, model_cam2 = load_model()
    matx = load_transformation_matrix()
    perform_segmentation(
        encoder_cam1,
        model_cam1,
        encoder_cam2,
        model_cam2,
        matx,
        label,
    )


@app.command()
def convert_to_reflectance(panel_reflectance: float):
    convert_images_to_reflectance(panel_reflectance)


@app.command()
def create_signatures(average_pixels: bool = True):
    create_spectral_signatures(average_pixels)


if __name__ == "__main__":
    logger.info(f"Project name: '{settings.project_name}'")
    app()
