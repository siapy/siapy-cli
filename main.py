import typer

from source.core import logger, settings
from source.misc.check_images import check_spectral_images
from source.misc.display_image import display_spectral_image

app = typer.Typer()


@app.command()
def main(name: str, lastname: str):
    logger.debug(settings)
    print(f"Hello {name} {lastname}")


@app.command()
def check_images():
    check_spectral_images()


@app.command()
def display_image(image_name: str):
    display_spectral_image()


if __name__ == "__main__":
    logger.info(f"Project name: '{settings.project_name}'")
    app()
