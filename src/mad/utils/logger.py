from loguru import logger
import sys

_inactive_sources: set[str] = set()
_active_sources: set[str] = set()
_all_disabled: bool = False

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
    if _all_disabled:
        return False
    source = record["extra"].get("source", "Unknown")
    if _active_sources:
        return source in _active_sources
    return source not in _inactive_sources


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
        filter=_source_filter,
    )

    return logger


def configure_logger(
    inactive_sources: list[str] | None = None,
    active_sources: list[str] | None = None,
    disable_all: bool = False,
):
    global _inactive_sources, _active_sources, _all_disabled
    _inactive_sources = set(inactive_sources or [])
    _active_sources = set(active_sources or [])
    _all_disabled = disable_all
    logger.remove()
    logger.add(
        sys.stdout,
        format=formatter,
        colorize=True,
        filter=_source_filter,
    )


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

    def success(self, message, *a, **kw):
        self._logger.success(message, *a, **kw)


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
