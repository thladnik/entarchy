"""Reading and writing the ASDF block files an exported archive is made of.

The value codec lives in `blob_store`: an archive stores the same encoded tree,
with the plain data as YAML and the arrays as ASDF blocks. What is here is the
file handling - the open-file cache, and resolving an "asdf:<file>#<key>"
pointer to a value.

ASDF is used for archives and not for live storage because its per-file header
costs about 580 bytes and 17 ms. A group file holds thousands of blobs, so that
is paid once; one attribute value per file would pay it every time.
"""
from __future__ import annotations

import collections
from typing import Any

from .blob_store import EncodingReport, decode, encode  # noqa: F401  (archive tooling)
from .blob_store import ENCODING_VERSION, TYPE_KEY  # noqa: F401

# Pointer form for values held in an archive: "asdf:<file>#<key>", with the file
#  given relative to the entarchy root
STORE_PREFIX = 'asdf:'


def make_store(relative_path: str, key: str) -> str:
    return f'{STORE_PREFIX}{relative_path}#{key}'


def parse_store(store: str) -> tuple[str, str]:
    relative_path, _, key = store[len(STORE_PREFIX):].partition('#')
    return relative_path, key


# Open ASDF files, kept per process. Reading an attribute must not reopen and
#  reparse the group file every time; opening one measured 0.2-2 s depending on
#  how many entries its tree holds, against 0.3 ms for the array read itself.
#
# The limit has to exceed the number of groups a read pass touches, or every
# read evicts a file another read is about to want. Measured over 12 groups:
# 1.31 ms per entity from a live entarchy, 1.93 ms from an archive with room for
# all 12, and 16.94 ms with room for 8. Open files hold a parsed tree and a file
# handle, both small for a group file, so the default is generous.

_OPEN_FILES: collections.OrderedDict = collections.OrderedDict()
_OPEN_FILE_LIMIT = 64


def set_open_file_limit(limit: int) -> None:
    """Set how many archive files stay open per process."""
    global _OPEN_FILE_LIMIT

    if limit < 1:
        raise ValueError('open file limit must be at least 1')

    _OPEN_FILE_LIMIT = limit
    while len(_OPEN_FILES) > _OPEN_FILE_LIMIT:
        _, evicted = _OPEN_FILES.popitem(last=False)
        try:
            evicted.close()
        except Exception:
            pass


def open_asdf_file(path: str, memmap: bool = False):
    """Open an ASDF file, reusing an already open handle where possible."""
    import asdf

    key = str(path)
    if key in _OPEN_FILES:
        _OPEN_FILES.move_to_end(key)
        return _OPEN_FILES[key]

    while len(_OPEN_FILES) >= _OPEN_FILE_LIMIT:
        _, evicted = _OPEN_FILES.popitem(last=False)
        try:
            evicted.close()
        except Exception:
            pass

    # lazy_load defers block reads, so opening a group file does not pull in every
    #  array it holds. memmap is off by default: memory mapped arrays stop being
    #  readable once their file is closed, and files here are closed on eviction.
    handle = asdf.open(key, mode='r', lazy_load=True, memmap=memmap)
    _OPEN_FILES[key] = handle

    return handle


def close_asdf_files() -> int:
    """Close every cached handle. Returns how many were open."""
    count = len(_OPEN_FILES)
    while len(_OPEN_FILES) > 0:
        _, handle = _OPEN_FILES.popitem()
        try:
            handle.close()
        except Exception:
            pass

    return count


def read_blob(root_path: str, relative_path: str, key: str, memmap: bool = False) -> Any:
    """Read one encoded attribute out of an archive block file."""
    import os

    handle = open_asdf_file(os.path.join(str(root_path), relative_path), memmap=memmap)

    try:
        node = handle['blobs'][key]
    except KeyError:
        raise KeyError(f'Archive file "{relative_path}" holds no blob "{key}". '
                       f'The archive may be incomplete.')

    return decode(node)


def available() -> bool:
    """Whether the asdf package is installed."""
    try:
        import asdf  # noqa: F401
    except ImportError:
        return False

    return True


def require() -> None:
    if not available():
        raise ImportError('This requires the asdf package. Install it with '
                          '"pip install entarchy[asdf]" or "pip install asdf".')
