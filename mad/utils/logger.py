"""Logur based logger for MAD simulation.
The user can define which sources are active or inactive, and the logger will filter messages accordingly.
The logger is configured to output to stdout with a custom format that includes the time, level, source, and message.
The source is color-coded based on the source name.
"""

from loguru import logger
import sys

_STATE = {
    "inactive_sources": set(),
    "active_sources": set(),
    "disable_all": False,
}
_SINK = {
    "handler_id": None,
    "default_removed": False,
}

# Definition of colors for the different sources.
# If unspecified, the default color is yellow.
# See https://loguru.readthedocs.io/en/stable/api/logger.html#color
SOURCE_COLORS = {
    "Simulation": "<white>",
    "Missile": "<red>",
    "Rocket": "<red>",
    "Interceptor": "<green>",
    "Physics": "<white>",
    "Projectile": "<blue>",
    "Unknown": "<yellow>",
    "I/O": "<cyan>",
    "Satellite": "<magenta>",
}


def _source_filter(record) -> bool:
    if _STATE["disable_all"]:
        return False
    source = record["extra"].get("source", "Unknown")
    if _STATE["active_sources"]:
        return source in _STATE["active_sources"]
    return source not in _STATE["inactive_sources"]


def formatter(record):
    source = record["extra"].get("source", "Unknown")
    color = SOURCE_COLORS.get(source, "<yellow>")

    return "<green>{time:HH:mm:ss}</green> | " "<level>{level:<8}</level> | " f"{color}{source:<12}</> | " "{message}\n"


def get_logger():
    if not _SINK["default_removed"]:
        # Loguru's default stderr sink (id 0) would otherwise duplicate our output.
        try:
            logger.remove(0)
        except ValueError:
            pass
        _SINK["default_removed"] = True
    if _SINK["handler_id"] is not None:
        logger.remove(_SINK["handler_id"])
    _SINK["handler_id"] = logger.add(
        sys.stdout,
        format=formatter,
        colorize=True,
        filter=_source_filter,
    )

    return logger


def configure_logger(
    inactive_sources: list[str] | None = None,
    active_sources: list[str] | None = None,
    disable_all: bool = False,
):
    # _source_filter reads _STATE on every record, so no handler rebuild is needed.
    _STATE["inactive_sources"] = set(inactive_sources or [])
    _STATE["active_sources"] = set(active_sources or [])
    _STATE["disable_all"] = disable_all


class SourceLogger:
    def __init__(self, base_logger=None, source: str | None = None):
        self._logger = base_logger or get_logger()
        if source is not None:
            self._logger = self._logger.bind(source=source)

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

    def success(self, message, *a, **kw):
        self._logger.success(message, *a, **kw)

    def __getitem__(self, source: str):
        return SourceLogger(self._logger, source)


if __name__ == "__main__":

    madlogger = SourceLogger()
    madlogger["Simulation"].warning("test")
    madlogger["Missile"].info("info")
    madlogger["dfdsf"].critical("critical")

    print("--- Missile silenced ---")
    configure_logger(inactive_sources=["Missile"])
    madlogger["Simulation"].warning("test")
    madlogger["Missile"].info("this should not appear")

    print("--- Only Simulation active ---")
    configure_logger(active_sources=["Simulation"])
    madlogger["Simulation"].warning("visible")
    madlogger["Missile"].info("this should not appear")
    madlogger["Projectile"].debug("this should not appear")

    print("--- All disabled ---")
    configure_logger(disable_all=True)
    madlogger["Simulation"].warning("this should not appear")
