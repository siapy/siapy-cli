import typer

from source.core import logger, settings


def main(name: str, lastname: str):
    logger.debug(settings)
    print(f"Hello {name} {lastname}")


if __name__ == "__main__":
    typer.run(main)
