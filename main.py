import typer

from source.core import logger, settings
from source.misc.check_images import check_spectral_images

app = typer.Typer()


@app.command()
def main(name: str, lastname: str):
    logger.debug(settings)
    print(f"Hello {name} {lastname}")


@app.command()
def check_images():
    check_spectral_images()


if __name__ == "__main__":
    app()
