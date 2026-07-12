"""Open the Studio UI as a chromeless desktop window via a Chromium browser's app mode.

Zero new dependencies: we find an installed Edge/Chrome/Chromium and launch it with
``--app=<url>`` and a throwaway profile dir, which renders a single, address-bar-less window.
Because we own the subprocess handle, the caller can wait on it to implement "close the window
to stop the app". On Windows 11 Edge is always present, so this works out of the box.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Well-known install locations, checked before PATH for the common desktop case.
_WINDOWS_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
)
_MACOS_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)
_PATH_NAMES = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "msedge", "chrome")


def find_chromium() -> str | None:
    """Return a path to an installed Chromium-family browser, or ``None``."""
    candidates: tuple[str, ...] = ()
    if sys.platform.startswith("win"):
        candidates = _WINDOWS_CANDIDATES
    elif sys.platform == "darwin":
        candidates = _MACOS_CANDIDATES
    for path in candidates:
        if os.path.exists(path):
            return path
    for name in _PATH_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def open_app_window(
    url: str,
    *,
    profile_dir: Path | None = None,
    width: int = 1440,
    height: int = 900,
) -> subprocess.Popen | None:
    """Launch the UI in a chromeless app window. Returns the process handle, or ``None`` if
    no Chromium browser is installed (caller should fall back to a normal browser tab)."""
    browser = find_chromium()
    if browser is None:
        return None
    if profile_dir is None:
        profile_dir = Path(tempfile.mkdtemp(prefix="nar-studio-profile-"))
    args = [
        browser,
        f"--app={url}",
        f"--user-data-dir={profile_dir}",
        f"--window-size={width},{height}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    return subprocess.Popen(args)
