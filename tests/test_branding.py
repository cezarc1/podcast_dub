"""Terminal branding behavior."""

import logging

from podcast_dub.branding import BANNER, show_banner


def test_show_banner_logs_complete_wordmark_once(caplog) -> None:
    with caplog.at_level(logging.INFO, logger="podcast_dub.branding"):
        show_banner()

    assert len(caplog.records) == 1
    assert caplog.records[0].getMessage() == f"\n{BANNER}"
    assert "____" in BANNER
