import importlib.util
import json
import pickle
import sys
from collections import OrderedDict
from collections.abc import Generator
from pathlib import Path
from typing import Any

from source.core import logger


def read_json(fname: str | Path) -> OrderedDict:
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Any, fname: str | Path) -> None:
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_pickle(content: Any, fname: str | Path) -> None:
    fname = Path(fname)
    with open(fname, "wb") as f:
        pickle.dump(content, f)


def read_pickle(fname: str | Path) -> Any:
    fname = Path(fname)
    with open(fname, "rb") as f:
        return pickle.load(f)


def write_txt(content: str, fname: str | Path) -> None:
    fname = Path(fname)
    with fname.open("w") as handle:
        handle.write(content)


def read_txt(fname: str | Path) -> str:
    fname = Path(fname)
    with fname.open("r") as handle:
        return handle.read()


def dict_zip(*dicts: dict[str, Any]) -> Generator[tuple[str, Any, Any], None, None]:
    if not dicts:
        return

    n = len(dicts[0])
    if any(len(d) != n for d in dicts):
        raise ValueError("Arguments must have the same length.")

    for key, first_val in dicts[0].items():
        yield key, first_val, *(other[key] for other in dicts[1:])


def import_class(
    class_name: str | None,
    directory: Path,
    default_class_name: str,
) -> Any:
    if class_name is None:
        class_name = default_class_name

    logger.info(f"Searching for class '{class_name}' in directory '{directory}'")

    # Add the directory to the system path if it's not already there
    if str(directory) not in sys.path:
        sys.path.append(str(directory))

    for file in directory.glob("*.py"):
        module_name = file.stem
        logger.info(f"Attempting to load module '{module_name}' from file '{file}'")
        spec = importlib.util.spec_from_file_location(module_name, file)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if hasattr(module, class_name):
            logger.info(f"Loading class '{class_name}' from module '{module_name}'")
            return getattr(module, class_name)()
    raise FileNotFoundError(
        f"Class '{class_name}' not found in any module in '{directory}'"
    )
