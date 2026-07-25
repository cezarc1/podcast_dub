"""Terminal branding for CLI startup."""

from __future__ import annotations

import logging
from typing import Final

logger = logging.getLogger(__name__)

BANNER: Final[str] = r"""
                     __                __            __      __
    ____  ____  ____/ /________ ______/ /_      ____/ /_  __/ /_
   / __ \/ __ \/ __  / ___/ __ `/ ___/ __/_____/ __  / / / / __ \
  / /_/ / /_/ / /_/ / /__/ /_/ (__  ) /_/_____/ /_/ / /_/ / /_/ /
 / .___/\____/\__,_/\___/\__,_/____/\__/      \__,_/\__,_/_.___/
/_/
""".strip("\n")


def show_banner() -> None:
    """Log the startup wordmark as one multiline record."""
    logger.info("\n%s", BANNER)
