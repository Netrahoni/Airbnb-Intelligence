"""Small, side-effect-free data-loading helpers shared by the dashboard and tests."""

from pathlib import Path
import pandas as pd


def file_cache_key(path):
    try:
        stat = Path(path).stat()
    except FileNotFoundError:
        return None
    return (stat.st_mtime_ns, stat.st_size)


def optional_file_cache_keys(filenames, search_dirs):
    return tuple(
        (str(Path(folder) / filename), file_cache_key(Path(folder) / filename))
        for filename in filenames
        for folder in search_dirs
    )


def read_csv_if_exists(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def read_optional_csv(filenames, search_dirs, on_error=None):
    for filename in filenames:
        for folder in search_dirs:
            path = Path(folder) / filename
            if not path.exists():
                continue
            try:
                return pd.read_csv(path)
            except Exception as error:
                if on_error is not None:
                    on_error(filename, error)
    return pd.DataFrame()


def read_report(path):
    return read_csv_if_exists(path)
