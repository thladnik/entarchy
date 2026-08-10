"""How an attribute value that is not a scalar becomes bytes, and back.

Two layers:

**The codec** (`encode` / `decode`) turns a Python value into a tree of plain
data plus numpy arrays. Ragged collections of arrays are concatenated into one
array with an offsets array beside them; tuples, bytes and numpy scalars are
tagged, because nothing else distinguishes them from lists, arrays and Python
scalars. Anything that cannot be expressed - custom classes, object arrays,
pandas objects - falls back to a pickled block, and `encode` records every such
fallback in the report it is given.

**The container** (`dumps` / `loads`) packs that tree into one byte string: a
JSON header describing the structure, followed by the arrays as raw C-order
buffers. Their lengths are implied by the dtype and shape in the header, so the
buffers need no framing of their own.

The point of the container is what it *cannot* say. A pickle is a program for a
stack machine - `STACK_GLOBAL` imports an arbitrary module attribute and
`REDUCE` calls it - so reading one executes code, and the bytes name the classes
they were written from. Pickling a list of arrays embeds
`numpy._core.multiarray._reconstruct`, a private path that numpy renamed in 2.0
and now carries a shim for. This header has a closed vocabulary instead: the
kinds below, JSON scalars, lists, mappings, and array references carrying a
dtype and a shape. `loads` is a dispatch over that vocabulary and has no
construct that can import or call anything, so malformed input produces a
ValueError rather than an import.

The same codec feeds the ASDF archive format, where the tree becomes a YAML tree
and the arrays become ASDF blocks. ASDF is not used here: its per-file header
costs about 580 bytes and 17 ms, which is amortised over the thousands of blobs
an archive group file holds but not over one attribute value.
"""
from __future__ import annotations

import collections
import json
import math
import pickle
import zlib
from typing import Any, Union

import numpy as np

# Marks a mapping as an encoded value rather than a plain dict. Chosen to be
#  unlikely as a real dict key; a value that happens to use it is still handled,
#  by encoding the dict explicitly (see _encode_dict).
TYPE_KEY = '__entarchy__'

# Bumped when the encoding changes in a way older readers cannot handle
ENCODING_VERSION = 1

MAGIC = b'ENTB'
CONTAINER_VERSION = 1

CODEC_NONE = 0
CODEC_ZLIB = 1

# Compression is decided per value, because it is worth 5x on the index-list
# shaped attributes and nothing at all on float traces. Trying it and keeping
# the result only when it pays means an incompressible value is stored raw and
# costs nothing to read.
#
# Level 1 rather than 6: measured over 350 real attribute values it reaches the
# same total size (65% of the pickle they replace) for two thirds of the write
# cost. Level 9 buys another 1% for five times the write cost.
COMPRESSION_LEVEL = 1
MIN_SIZE_TO_COMPRESS = 512
COMPRESSION_MUST_SAVE = 0.9


class EncodingReport:
    """Collects what had to be pickled, so a caller can report it."""

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


# ---------------------------------------------------------------------------
# The codec: Python value <-> tree of plain data and arrays
# ---------------------------------------------------------------------------


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
    data = as_array(data)
    return [data[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]


def as_array(value: Any) -> np.ndarray:
    """Materialise an array proxy into a real ndarray.

    asdf hands back NDArrayType rather than ndarray, which is close enough for
    arithmetic but not for `isinstance(value, np.ndarray)` or for anything that
    dispatches on the concrete type. Callers get the same array they stored, so
    the substitution never becomes visible.
    """
    if isinstance(value, np.ndarray):
        return value

    return np.asarray(value)


def _encode_dict(value: dict, report: EncodingReport, where: str) -> Any:
    # A mapping with well behaved keys stays a plain mapping, which is what makes
    #  the header readable. Anything else is made explicit.
    keys_are_plain = all(isinstance(key, str) and key != TYPE_KEY for key in value)
    if keys_are_plain:
        return {key: encode(item, report, f'{where}.{key}') for key, item in value.items()}

    return {
        TYPE_KEY: 'dict',
        'keys': [encode(key, report, f'{where}.<key>') for key in value],
        'values': [encode(item, report, f'{where}.<value>') for item in value.values()],
    }


def encode(value: Any, report: EncodingReport = None, where: str = '') -> Any:
    """Turn a Python value into a tree of plain data and numpy arrays."""
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
            return _unpack_array_list(node['data'], as_array(node['offsets']))

        if kind == 'ragged2':
            inner = _unpack_array_list(node['data'], as_array(node['offsets']))
            outer = as_array(node['outer_offsets'])
            return [inner[outer[i]:outer[i + 1]] for i in range(len(outer) - 1)]

        if kind == 'tuple':
            return tuple(decode(item) for item in node['items'])

        if kind == 'bytes':
            return as_array(node['data']).tobytes()

        if kind == 'npscalar':
            return np.dtype(node['dtype']).type(node['value'])

        if kind == 'dict':
            return {decode(key): decode(item)
                    for key, item in zip(node['keys'], node['values'])}

        if kind == 'pickle':
            return pickle.loads(as_array(node['data']).tobytes())

        raise ValueError(f'Unknown encoded value kind "{kind}". The data was written by a '
                         f'newer version of entarchy than this one.')

    if isinstance(node, list):
        return [decode(item) for item in node]

    # Plain arrays, and the proxies asdf returns in their place
    if isinstance(node, np.ndarray) or hasattr(node, '__array__'):
        return as_array(node)

    return node


# ---------------------------------------------------------------------------
# The container: tree -> bytes
# ---------------------------------------------------------------------------

# JSON has no way to write a non-finite float, and Python's json emits NaN and
#  Infinity, which is not JSON any more. Non-finite floats reaching the header
#  are rare - a plain float attribute goes to value_float, so this is only a
#  float inside a list or mapping - and cheap to name explicitly.
_FLOAT_KEY = '__f__'
_FLOAT_NAMES = {math.inf: 'inf', -math.inf: '-inf'}


def _to_header(node: Any, arrays: list, specs: list) -> Any:
    """Replace arrays with references and non-finite floats with names."""
    if isinstance(node, np.ndarray):
        contiguous = np.ascontiguousarray(node)
        arrays.append(contiguous)
        specs.append([np.lib.format.dtype_to_descr(contiguous.dtype),
                      list(contiguous.shape)])
        return {'__a__': len(arrays) - 1}

    if isinstance(node, float) and not math.isfinite(node):
        return {_FLOAT_KEY: _FLOAT_NAMES.get(node, 'nan')}

    if isinstance(node, dict):
        return {key: _to_header(item, arrays, specs) for key, item in node.items()}

    if isinstance(node, list):
        return [_to_header(item, arrays, specs) for item in node]

    return node


def _from_header(node: Any, arrays: list) -> Any:
    if isinstance(node, dict):
        if len(node) == 1:
            if '__a__' in node:
                return arrays[node['__a__']]
            if _FLOAT_KEY in node:
                return float(node[_FLOAT_KEY])

        return {key: _from_header(item, arrays) for key, item in node.items()}

    if isinstance(node, list):
        return [_from_header(item, arrays) for item in node]

    return node


def _descr_to_dtype(descr: Any) -> np.dtype:
    # JSON turns the tuples of a structured dtype's description into lists
    if isinstance(descr, list):
        descr = [tuple(field) if isinstance(field, list) else field for field in descr]

    return np.lib.format.descr_to_dtype(descr)


def _pack(header: dict, arrays: list, compress: bool) -> bytes:
    body = json.dumps(header, allow_nan=False, separators=(',', ':')).encode('utf-8')
    parts = [len(body).to_bytes(4, 'little'), body]
    parts.extend(array.tobytes() for array in arrays)
    payload = b''.join(parts)

    codec = CODEC_NONE
    if compress and len(payload) >= MIN_SIZE_TO_COMPRESS:
        squeezed = zlib.compress(payload, COMPRESSION_LEVEL)
        # Only keep it when it actually pays; float traces do not compress and
        #  would otherwise carry the cost of a zlib pass on every read
        if len(squeezed) < len(payload) * COMPRESSION_MUST_SAVE:
            payload, codec = squeezed, CODEC_ZLIB

    return b''.join([MAGIC, bytes([CONTAINER_VERSION, codec]), payload])


def _unpack(raw: bytes) -> tuple[dict, list]:
    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise ValueError(f'Blob data must be bytes, got {type(raw).__name__}.')

    raw = bytes(raw)
    if raw[:len(MAGIC)] != MAGIC:
        raise ValueError('Blob data does not start with the entarchy blob marker. '
                         'It was written by a different version of entarchy, or the '
                         'row is not a blob attribute.')

    version, codec = raw[len(MAGIC)], raw[len(MAGIC) + 1]
    if version != CONTAINER_VERSION:
        raise ValueError(f'Blob container version {version} cannot be read by this '
                         f'version of entarchy (which writes {CONTAINER_VERSION}).')

    payload = raw[len(MAGIC) + 2:]
    if codec == CODEC_ZLIB:
        payload = zlib.decompress(payload)
    elif codec != CODEC_NONE:
        raise ValueError(f'Unknown blob compression codec {codec}.')

    header_length = int.from_bytes(payload[:4], 'little')
    header = json.loads(payload[4:4 + header_length])

    arrays, position = [], 4 + header_length
    for descr, shape in header.get('a', []):
        dtype = _descr_to_dtype(descr)
        count = int(np.prod(shape)) if len(shape) > 0 else 1
        size = count * dtype.itemsize
        # A copy, not a view on the buffer: the value outlives the row it came
        #  from, and a read-only view of a bytes object cannot be written to
        arrays.append(np.frombuffer(payload, dtype=dtype, count=count,
                                    offset=position).reshape(shape).copy())
        position += size

    return header, arrays


def dumps(value: Any, report: EncodingReport = None, where: str = '',
          compress: bool = True) -> bytes:
    """Encode one attribute value into a self-contained byte string."""
    arrays, specs = [], []
    tree = _to_header(encode(value, report, where), arrays, specs)

    return _pack({'t': tree, 'a': specs}, arrays, compress)


def dumps_external(relative_path: str) -> bytes:
    """A pointer to a value held in a file of its own under the entarchy root."""
    return _pack({'x': relative_path}, [], compress=False)


def dumps_archived(relative_path: str, key: str) -> bytes:
    """A pointer to a value held in an archive's ASDF block file."""
    return _pack({'z': [relative_path, key]}, [], compress=False)


def store_of(raw: bytes) -> str:
    """Where the value actually lives: 'internal', a path, or 'asdf:<file>#<key>'."""
    header, _ = _unpack(raw)

    if 'x' in header:
        return header['x']
    if 'z' in header:
        return f'asdf:{header["z"][0]}#{header["z"][1]}'

    return 'internal'


def loads(raw: bytes, root_path: Union[str, None] = None, memmap: bool = False) -> Any:
    """Rebuild an attribute value from its stored bytes.

    `root_path` is the entarchy directory, needed only when the value lives in a
    file of its own rather than in the row.
    """
    import os

    header, arrays = _unpack(raw)

    if 'x' in header:
        if root_path is None:
            raise ValueError(f'Value is stored in "{header["x"]}" but no entarchy path '
                             f'was given to resolve it against.')
        with open(os.path.join(str(root_path), header['x']), 'rb') as f:
            return loads(f.read(), root_path=root_path, memmap=memmap)

    if 'z' in header:
        from . import asdf_store

        if root_path is None:
            raise ValueError(f'Value is stored in archive file "{header["z"][0]}" but no '
                             f'entarchy path was given to resolve it against.')
        relative_path, key = header['z']
        return asdf_store.read_blob(root_path, relative_path, key, memmap=memmap)

    return decode(_from_header(header['t'], arrays))
