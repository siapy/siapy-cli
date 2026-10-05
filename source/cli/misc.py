import typer
from source.misc.check_images import check_spectral_images
from source.misc.display_image import (
    display_spectral_image,
)

app = typer.Typer()


@app.command()
def check_images():
    check_spectral_images()


@app.command()
def display_image(label: str | None = None):
    display_spectral_image(label)
