import sys
from importlib.util import module_from_spec, spec_from_file_location
from typing import Optional

from source.analysis.base import BaseDataLoader
from source.core import logger, settings

_DATA_LOADERS_DIR = settings.extensions_dir / "data_loaders"


def import_data_loader(data_loader_name: Optional[str]) -> BaseDataLoader:
    if data_loader_name is None:
        data_loader_name = "DataLoaderExample"

    for file in _DATA_LOADERS_DIR.glob("*.py"):
        spec = spec_from_file_location(file.stem, file)
        module = module_from_spec(spec)
        sys.modules[file.stem] = module
        spec.loader.exec_module(module)
        if hasattr(module, data_loader_name):
            logger.info(f"Loading data loader: '{file}'")
            return getattr(module, data_loader_name)()
    raise FileNotFoundError(
        f"Data loader class '{data_loader_name}' not found in any module in '{_DATA_LOADERS_DIR}'"
    )
