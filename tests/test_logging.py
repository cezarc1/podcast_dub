"""Logging configuration and print-free package surface."""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from podcast_dub.logging_config import configure_logging

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "src" / "podcast_dub"


def test_configure_logging_sets_info_message_only_format(capsys) -> None:
    configure_logging(force=True)
    logger = logging.getLogger("podcast_dub.test_logging")
    logger.info("hello progress")
    logger.debug("hidden debug")
    captured = capsys.readouterr()
    assert captured.err.strip() == "hello progress"
    assert captured.out == ""


def test_package_has_no_print_calls() -> None:
    offenders = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "print":
                offenders.append(f"{path.relative_to(PACKAGE_ROOT.parent.parent)}:{node.lineno}")
    assert offenders == [], "unexpected print() calls:\n" + "\n".join(offenders)
