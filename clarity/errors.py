"""Errors a person can read. They do not include a traceback."""
from __future__ import annotations


class ClarityError(RuntimeError):
    """The job cannot continue. ``str(error)`` is the message to show."""
