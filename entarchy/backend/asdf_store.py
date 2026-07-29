"""Translation between entarchy attribute values and ASDF tree nodes.

ASDF stores numpy arrays as binary blocks and everything else as YAML. Plain
Python containers - lists, dicts, strings, numbers - therefore need no help at
all. Two things do:

**Ragged collections of arrays.** A list of arrays becomes one binary block per
array. `bs_cluster_full_indices` in the CMN analysis is a list of 1000
bootstrap iterations, each a list of a handful of index arrays: measured at
2967 blocks, 1.2 MB and 1.7 s to write for a single ROI attribute, against
836 kB and 13 ms for the pickle it would replace. Concatenating the arrays and
keeping their boundaries in an offsets array brings that to 3 blocks, 767 kB
and 39 ms, with the values bit-identical.

**Types YAML does not have.** Tuples come back as lists and bytes have no
representation at all, so both are tagged explicitly.

Anything that cannot be expressed - custom classes, pandas objects, object
arrays - falls back to a pickled block. That keeps the export lossless, but a
file containing such a value can only be read where those classes are
importable, which defeats the point of an archive. `encode` records every
fallback in the report it is given so the exporter can list them.
"""
from __future__ import annotations

import collections
import pickle
from typing import Any, Union

import numpy as np

# Marks a mapping as an encoded value rather than a plain dict. Chosen to be
#  unlikely as a real dict key; a value that happens to use it is still handled,
#  by encoding the dict explicitly (see _encode_dict).
TYPE_KEY = '__entarchy__'

# Bumped when the encoding changes in a way older readers cannot handle
ENCODING_VERSION = 1

# Serializer._store form for values held in an archive: "asdf:<file>#<key>",
#  with the file given relative to the entarchy root
STORE_PREFIX = 'asdf:'


def make_store(relative_path: str, key: str) -> str:
    return f'{STORE_PREFIX}{relative_path}#{key}'


def parse_store(store: str) -> tuple[str, str]:
    relative_path, _, key = store[len(STORE_PREFIX):].partition('#')
    return relative_path, key


class EncodingReport:
    """Collects what had to be pickled, so the exporter can report it."""

    def __init__(self):
        self.pickled: list[tuple[str, str]] = []
        self.ragged_packed: int = 0

    def note_pickle(self, where: str, type_name: str) -> None:
        self.pickled.append((where, type_name))

    @property
    def is_fully_portable(self) -> bool:
        return len(self.pickled) == 0

    def summary(self) -> str:
        if self.is_fully_portable:
            return 'all values encoded natively'

        by_type = collections.Counter(type_name for _, type_name in self.pickled)
        parts = ', '.join(f'{count}x {name}' for name, count in by_type.most_common())
        return f'{len(self.pickled)} value(s) fell back to pickle: {parts}'


def _is_plain_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (bool, int, float, str))


def _packable_array_list(values: list) -> bool:
    """True if every element is an array that can share one concatenated block.

    Requires a common dtype and matching trailing dimensions, since the arrays
    are joined along axis 0. Object dtype cannot go into a block at all.
    """
    if len(values) == 0:
        return False

    first = values[0]
    if not isinstance(first, np.ndarray) or first.dtype == object or first.ndim == 0:
        return False

    dtype, tail = first.dtype, first.shape[1:]
    for value in values:
        if not isinstance(value, np.ndarray):
            return False
        if value.dtype != dtype or value.ndim != first.ndim or value.shape[1:] != tail:
            return False

    return True


def _pack_array_list(values: list) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate arrays into one block plus the offsets that split them again."""
    data = np.concatenate(values, axis=0) if len(values) > 0 else np.zeros(0)
    offsets = np.zeros(len(values) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(np.array([len(value) for value in values], dtype=np.int64))

    return data, offsets


def _unpack_array_list(data: np.ndarray, offsets: np.ndarray) -> list[np.ndarray]:
    data = _as_array(data)
    return [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def _as_array(value: Any) -> np.ndarray:
    """Materialise an ASDF array proxy into a real ndarray.

    asdf hands back NDArrayType rather than ndarray, which is close enough for
    arithmetic but not for `isinstance(value, np.ndarray)`, for pickling, or for
    anything that dispatches on the concrete type - including entarchy's own
    Serializer, which would fall through to pickling the proxy. Callers get the
    same array they stored, so the substitution never becomes visible.
    """
    if isinstance(value, np.ndarray):
        return value

    return np.asarray(value)


def _encode_dict(value: dict, report: EncodingReport, where: str) -> Any:
    # A mapping with well behaved keys stays a plain YAML mapping, which is what
    #  makes the file readable without entarchy. Anything else is made explicit.
    keys_are_plain = all(isinstance(key, str) and key != TYPE_KEY for key in value)
    if keys_are_plain:
        return {key: encode(item, report, f'{where}.{key}') for key, item in value.items()}

    return {
        TYPE_KEY: 'dict',
        'keys': [encode(key, report, f'{where}.<key>') for key in value],
        'values': [encode(item, report, f'{where}.<value>') for item in value.values()],
    }


def encode(value: Any, report: EncodingReport = None, where: str = '') -> Any:
    """Turn a Python value into something ASDF can write."""
    if report is None:
        report = EncodingReport()

    if _is_plain_scalar(value):
        return value

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            # Object arrays hold arbitrary Python; there is no block form
            report.note_pickle(where, 'ndarray[object]')
            return {TYPE_KEY: 'pickle', 'data': _pickle_to_array(value)}
        return value

    if isinstance(value, np.generic):
        # A numpy scalar would otherwise degrade to a Python float/int
        return {TYPE_KEY: 'npscalar', 'dtype': value.dtype.str, 'value': value.item()}

    if isinstance(value, bytes):
        return {TYPE_KEY: 'bytes', 'data': np.frombuffer(value, dtype=np.uint8)}

    if isinstance(value, tuple):
        return {TYPE_KEY: 'tuple',
                'items': [encode(item, report, f'{where}[]') for item in value]}

    if isinstance(value, list):
        return _encode_list(value, report, where)

    if isinstance(value, dict):
        return _encode_dict(value, report, where)

    report.note_pickle(where, type(value).__name__)
    return {TYPE_KEY: 'pickle', 'data': _pickle_to_array(value)}


def _encode_list(value: list, report: EncodingReport, where: str) -> Any:
    if _packable_array_list(value):
        data, offsets = _pack_array_list(value)
        report.ragged_packed += 1
        return {TYPE_KEY: 'ragged', 'data': data, 'offsets': offsets}

    # A list of lists of arrays - the bootstrap shape - packs one level deeper
    if len(value) > 0 and all(isinstance(item, list) for item in value):
        flat = [array for item in value for array in item]
        if _packable_array_list(flat):
            data, offsets = _pack_array_list(flat)
            outer = np.zeros(len(value) + 1, dtype=np.int64)
            outer[1:] = np.cumsum(np.array([len(item) for item in value], dtype=np.int64))
            report.ragged_packed += 1
            return {TYPE_KEY: 'ragged2', 'data': data,
                    'offsets': offsets, 'outer_offsets': outer}

    return [encode(item, report, f'{where}[]') for item in value]


def _pickle_to_array(value: Any) -> np.ndarray:
    return np.frombuffer(pickle.dumps(value), dtype=np.uint8)


def decode(node: Any) -> Any:
    """Rebuild the original Python value from an encoded tree node."""
    if isinstance(node, dict):
        kind = node.get(TYPE_KEY)

        if kind is None:
            return {key: decode(item) for key, item in node.items()}

        if kind == 'ragged':
            return _unpack_array_list(node['data'], _as_array(node['offsets']))

        if kind == 'ragged2':
            inner = _unpack_array_list(node['data'], _as_array(node['offsets']))
            outer = _as_array(node['outer_offsets'])
            return [inner[outer[i]:outer[i + 1]] for i in range(len(outer) - 1)]

        if kind == 'tuple':
            return tuple(decode(item) for item in node['items'])

        if kind == 'bytes':
            return _as_array(node['data']).tobytes()

        if kind == 'npscalar':
            return np.dtype(node['dtype']).type(node['value'])

        if kind == 'dict':
            return {decode(key): decode(item)
                    for key, item in zip(node['keys'], node['values'])}

        if kind == 'pickle':
            return pickle.loads(_as_array(node['data']).tobytes())

        raise ValueError(f'Unknown encoded value kind "{kind}". The archive was written by a '
                         f'newer version of entarchy than this one.')

    if isinstance(node, list):
        return [decode(item) for item in node]

    # Plain arrays, and the proxies asdf returns in their place
    if isinstance(node, np.ndarray) or hasattr(node, '__array__'):
        return _as_array(node)

    return node


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
