import typer
from source.misc.check_images import check_spectral_images
from source.misc.display_image import (
    display_spectral_image,
)
from typing import Optional

app = typer.Typer()


@app.command()
def check_images():
    check_spectral_images()


@app.command()
def display_image(label: Optional[str] = None):
    display_spectral_image(label)
