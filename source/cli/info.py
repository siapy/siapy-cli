import json
from importlib.metadata import PackageNotFoundError, version

import typer
from pydantic.json import pydantic_encoder
from source.core import logger, settings

app = typer.Typer()


@app.command()
def display_settings():
    logger.info(json.dumps(settings.model_dump(), default=pydantic_encoder, indent=4))


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
