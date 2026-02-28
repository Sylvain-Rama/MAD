from loguru import logger
import sys

SOURCE_COLORS = {
    "Simulation": "<black>",
    "Missile": "<red>",
    "Physics": "<green>",
    "Projectile": "<blue>",
    "Unknown": "<yellow>",
}


def formatter(record):
    source = record["extra"].get("source", "Unknown")
    color = SOURCE_COLORS.get(source, "<yellow>")

    return "<green>{time:HH:mm:ss}</green> | " "<level>{level:<8}</level> | " f"{color}{source:<12}</> | " "{message}\n"


def get_logger():
    logger.remove()
    logger.add(
        sys.stdout,
        format=formatter,
        colorize=True,
    )

    return logger


class _BoundSourceLogger:

    def __init__(self, base_logger, source: str):
        self._logger = base_logger.bind(source=source)

    def debug(self, message, *a, **kw):
        self._logger.debug(message, *a, **kw)

    def info(self, message, *a, **kw):
        self._logger.info(message, *a, **kw)

    def warning(self, message, *a, **kw):
        self._logger.warning(message, *a, **kw)

    def error(self, message, *a, **kw):
        self._logger.error(message, *a, **kw)

    def critical(self, message, *a, **kw):
        self._logger.critical(message, *a, **kw)


class SourceLogger:

    def __init__(self, base_logger=get_logger()):
        self._logger = base_logger

    def __getitem__(self, source: str):
        return _BoundSourceLogger(self._logger, source)


if __name__ == "__main__":

    madlogger = SourceLogger()
    madlogger["Simulation"].warning("test")
    madlogger["Missile"].info("info")
    madlogger["dfdsf"].critical("critical")
