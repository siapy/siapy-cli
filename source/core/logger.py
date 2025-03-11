import logging
import sys
from functools import lru_cache

import loguru
from loguru import logger as _logger


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        logger_opt = _logger.opt(depth=7, exception=record.exc_info)
        logger_opt.log(record.levelname, record.getMessage())


@lru_cache()
def setup_logger(debug: bool = False) -> loguru._Logger:  # type: ignore
    LOGGING_LEVEL = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        handlers=[InterceptHandler(level=LOGGING_LEVEL)],
        level=LOGGING_LEVEL,
    )
    _logger.configure(handlers=[{"sink": sys.stderr, "level": LOGGING_LEVEL}])
    return _logger
