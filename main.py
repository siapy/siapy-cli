import json
from typing import Optional

import typer
from pydantic.json import pydantic_encoder

from source.core import logger, settings
from source.misc import check_spectral_images, display_spectral_image

app = typer.Typer()


@app.command()
def display_settings():
    logger.info(json.dumps(settings.model_dump(), default=pydantic_encoder, indent=4))


@app.command()
def check_images():
    check_spectral_images()


@app.command()
def display_image(image_name: Optional[str] = None):
    display_spectral_image(image_name)


if __name__ == "__main__":
    logger.info(f"Project name: '{settings.project_name}'")
    app()
