from __future__ import annotations

import os
import sys
from typing import Any, Iterable, Optional

from tqdm.auto import tqdm


_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "")
    return str(value).strip().lower() in _TRUE_VALUES


def progress_enabled(enabled: Optional[bool] = None) -> bool:
    """Return whether terminal progress should be shown."""
    if _env_flag("DISABLE_PROGRESS"):
        return False
    if enabled is False:
        return False
    if _env_flag("FORCE_PROGRESS"):
        return True
    return bool(sys.stderr.isatty())


def iter_progress(
    iterable: Iterable[Any],
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    enabled: Optional[bool] = None,
    leave: bool = False,
    mininterval: float = 0.2,
    **kwargs: Any,
):
    """Wrap iterable with tqdm, auto-disabled for non-interactive envs."""
    return tqdm(
        iterable,
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=float(mininterval),
        disable=not progress_enabled(enabled),
        file=sys.stderr,
        **kwargs,
    )


def bar_progress(
    *,
    total: Optional[int] = None,
    desc: Optional[str] = None,
    unit: Optional[str] = None,
    enabled: Optional[bool] = None,
    leave: bool = False,
    mininterval: float = 0.2,
    **kwargs: Any,
):
    """Create manual tqdm bar for update()/set_postfix() usage."""
    return tqdm(
        total=total,
        desc=desc,
        unit=unit,
        leave=leave,
        dynamic_ncols=True,
        mininterval=float(mininterval),
        disable=not progress_enabled(enabled),
        file=sys.stderr,
        **kwargs,
    )

