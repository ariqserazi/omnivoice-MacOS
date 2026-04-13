"""Application paths for OmniVoice local state on macOS and other platforms."""

from __future__ import annotations

import os
import platform
from pathlib import Path


def get_app_support_dir() -> Path:
    """Return the base directory for local OmniVoice app data."""
    override = os.environ.get("OMNIVOICE_APP_HOME")
    if override:
        return Path(override).expanduser().resolve()

    home = Path.home()
    if platform.system() == "Darwin":
        return home / "Library" / "Application Support" / "OmniVoice"
    if platform.system() == "Windows":
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "OmniVoice"
    return home / ".local" / "share" / "omnivoice"


def get_voice_library_dir() -> Path:
    """Return the directory where reusable voice profiles are stored."""
    override = os.environ.get("OMNIVOICE_VOICES_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return get_app_support_dir() / "voices"


def get_diagnostics_dir() -> Path:
    """Return the directory for cached device diagnostics."""
    return get_app_support_dir() / "diagnostics"
