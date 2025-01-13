import typer
from source.cli import analysis, info, misc, segment
from source.core import logger, settings

app = typer.Typer()
app.add_typer(info.app, name="info")
app.add_typer(misc.app, name="misc")
app.add_typer(segment.app, name="segment")
app.add_typer(analysis.app, name="analysis")


if __name__ == "__main__":
    logger.info(f"Project name: '{settings.project_name}'")
    app()
