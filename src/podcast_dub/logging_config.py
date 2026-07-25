"""Shared logging setup for package entry points."""

from __future__ import annotations

import logging
import sys


def configure_logging(*, level: int = logging.INFO, force: bool = True) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
        force=force,
    )
