"""Configuracion del Logging de la aplicacion"""

import logging

try:
    from rich.logging import RichHandler
except ImportError:
    RichHandler = None

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def resolved_level(nivel: str) -> int:
    return getattr(logging, nivel.upper(), logging.INFO)

def setup_logger(nivel: str = "INFO") -> None:
    
    root_logger = logging.getLogger()
    nivel_de_resolucion = nivel_de_resolucion(nivel)

    if root_logger.handlers:
        root_logger.setLevel(resolved_level)
        return

    if RichHandler is not None:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
        logging.basicConfig(
            level=resolved_level,
            format="%(message)s",
            datefmt=DEFAULT_DATE_FORMAT,
            handlers=[handler],
        )
        return

    logging.basicConfig(
        level=resolved_level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger by module name."""
    return logging.getLogger(name)

