import json
from typing import Optional

import typer
from pydantic.json import pydantic_encoder

from source.core import logger, settings
from source.helpers import save_selected_areas, save_transformation_matrix
from source.misc import check_spectral_images, display_spectral_image
from source.processing import find_transformation_between_images, select_areas_on_images

app = typer.Typer()


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
    save_selected_areas(selected_areas, category, label)


if __name__ == "__main__":
    logger.info(f"Project name: '{settings.project_name}'")
    app()
