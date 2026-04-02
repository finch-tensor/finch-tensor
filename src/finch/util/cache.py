# AI modified: 2026-04-02T20:46:24Z parent=154b5aeaa66d01a2373296ba9af9705a3db73ed9
import atexit
import shutil
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path
from uuid import UUID

import finchlite

from .config import config, get_version

finch_uuid = UUID("ef66f312-ff6e-4b8a-bb8c-9a843f3ecdf4")
cache_timestamp_filename = ".finch_code_mtime_ns"
_finch_source_root = Path(finchlite.__path__[0])


def _latest_finch_code_mtime_ns() -> int:
    latest_mtime = 0
    for path in _finch_source_root.rglob("*"):
        if (
            "__pycache__" not in path.parts
            and path.is_file()
            and path.suffix not in {".pyc", ".pyo"}
        ):
_checked_cache_roots: set[Path] = set()


def _latest_finch_code_mtime_ns() -> int:
    finch_root = Path(__file__).resolve().parents[1]
    latest_mtime = 0
    for path in finch_root.rglob("*"):
        if path.is_file():
            latest_mtime = max(latest_mtime, path.stat().st_mtime_ns)
    return latest_mtime


_session_finch_code_mtime_ns = _latest_finch_code_mtime_ns()
_cache_checked = False


def _clear_cache_root(cache_root: Path, *, keep_timestamp: bool = True) -> None:
    for path in cache_root.iterdir():
        if keep_timestamp and path.name == cache_timestamp_filename:
            continue
def _clear_cache_root(cache_root: Path) -> None:
    for path in cache_root.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def clear_cache() -> None:
    """Clear Finch's persistent cache for the current Finch version.

    This removes all cached files for the active Finch version under
    ``<data_path>/cache/<version>``. If the cache directory does not exist,
    this function does nothing.
    """

    global _cache_checked
    cache_root = Path(config.get("data_path")) / "cache" / get_version()
    if cache_root.exists():
        _clear_cache_root(cache_root, keep_timestamp=False)
    _cache_checked = False


def _ensure_cache_fresh(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    timestamp_file = cache_root / cache_timestamp_filename
    current_mtime = _session_finch_code_mtime_ns
    should_clear = False
def _ensure_cache_fresh(cache_root: Path) -> None:
    if cache_root in _checked_cache_roots:
        return
    _checked_cache_roots.add(cache_root)

    cache_root.mkdir(parents=True, exist_ok=True)
    timestamp_file = cache_root / cache_timestamp_filename
    current_mtime = _latest_finch_code_mtime_ns()

    if timestamp_file.exists():
        try:
            cached_mtime = int(timestamp_file.read_text().strip())
        except ValueError:
            should_clear = True
        else:
            should_clear = current_mtime > cached_mtime
    else:
        should_clear = True

    if should_clear:
        _clear_cache_root(cache_root)
            cached_mtime = -1
        if current_mtime > cached_mtime:
            _clear_cache_root(cache_root)

    timestamp_file.write_text(str(current_mtime))


def file_cache(*, ext: str, domain: str) -> Callable:
    """Caches the result of a function to a file.

    Args:
        ext: The file extension for the cache file.
        domain: The domain name for the cache file.

    Returns:
        A wrapper function that caches the result of the original function.
    """

    def decorator(f: Callable) -> Callable:
        nonlocal domain
        nonlocal ext
        ext = ext.lstrip(".")
        if config.get("cache_enable"):
            global _cache_checked
            cache_root = Path(config.get("data_path")) / "cache" / get_version()
            if not _cache_checked:
                _ensure_cache_fresh(cache_root)
                _cache_checked = True
            cache_root = Path(config.get("data_path")) / "cache" / get_version()
            _ensure_cache_fresh(cache_root)
            cache_dir = cache_root / domain
        else:
            cache_dir = Path(
                tempfile.mkdtemp(
                    prefix=str(Path(config.get("data_path")) / "tmp" / domain)
                )
            )
            atexit.register(
                lambda: shutil.rmtree(cache_dir) if cache_dir.exists() else None
            )

        cache_dir.mkdir(parents=True, exist_ok=True)

        def inner(*args):
            id = uuid.uuid5(finch_uuid, str((f.__name__, f.__module__, args)))
            filename = cache_dir / f"{f.__name__}_{id}.{ext}"
            if not config.get("cache_enable") or not filename.exists():
                f(str(filename), *args)
            return filename

        return inner

    return decorator
