"""Logging setup.

Rich handler for humans at the terminal, plain records everywhere else so log
output stays greppable when a run is piped to a file.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys

from rich.console import Console
from rich.logging import RichHandler

_CONFIGURED = False


def force_utf8_streams() -> None:
    """Make stdout/stderr able to carry the characters real documents contain.

    On Windows the console defaults to a legacy code page (cp1252), so printing
    a character a PDF actually used -- U+2217 from a paper's author footnote,
    say -- raises UnicodeEncodeError and kills the command. Since this tool
    exists to display extracted document text, that is not an edge case.

    ``errors="replace"`` rather than "strict": a character that cannot be shown
    should degrade to a placeholder, never abort the run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # A detached or already-wrapped stream cannot be reconfigured; that
            # is fine, it just keeps whatever encoding it has.
            with contextlib.suppress(ValueError, OSError):  # pragma: no cover
                reconfigure(encoding="utf-8", errors="replace")


def setup_logging(level: str | int | None = None, *, force: bool = False) -> None:
    """Configure the root logger once per process."""
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    force_utf8_streams()

    resolved = level or os.environ.get("MMRAG_LOG_LEVEL", "INFO")
    if isinstance(resolved, str):
        resolved = getattr(logging, resolved.upper(), logging.INFO)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(resolved)

    if sys.stderr.isatty():
        handler: logging.Handler = RichHandler(
            console=Console(stderr=True),
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
        )
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-8s %(name)s | %(message)s")
        )

    root.addHandler(handler)

    # These are chatty at INFO and drown out our own progress output.
    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
