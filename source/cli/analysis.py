from typing import Optional

import typer
from rich import print
from source.analysis.extensions import import_data_loader

app = typer.Typer()


@app.command()
def test_load_data(
    data_loader: Optional[str] = None,
):
    loader = import_data_loader(data_loader)
    loader.load_data()
    print(
        f"Signatures shape: {loader.load_data()[0].shape},"
        f"\nTargets shape: {loader.load_data()[1].shape}"
        f"\nTargets unique values: {list(set(loader.load_data()[1]))}"
    )
