"""How a non-scalar attribute value becomes bytes and comes back.

The point of the container is what it cannot do, so most of this is about
values surviving unchanged and about malformed input being refused rather than
executed.
"""
import json
import math
import re

import numpy as np
import pytest

from entarchy.backend import blob_store
from entarchy.backend.sql import _get_attribute_fp, _get_namehash


def same(a, b):
    """Equal in value, type and dtype - a round trip may not quietly convert."""
    if isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
                and np.array_equal(a, b, equal_nan=b.dtype.kind == 'f'))
    if isinstance(b, np.generic):
        return isinstance(a, np.generic) and a.dtype == b.dtype and a == b
    if isinstance(b, (list, tuple)):
        return (type(a) is type(b) and len(a) == len(b)
                and all(same(x, y) for x, y in zip(a, b)))
    if isinstance(b, dict):
        return (isinstance(a, dict) and a.keys() == b.keys()
                and all(same(a[key], b[key]) for key in b))
    if isinstance(b, float) and math.isnan(b):
        return isinstance(a, float) and math.isnan(a)
    return type(a) is type(b) and a == b


VALUES = {
    'float array': np.random.default_rng(0).random(200).astype(np.float32),
    'two dimensional': np.arange(24).reshape(4, 6),
    'bool array': np.arange(20) % 2 == 0,
    'string array': np.array(['a', 'bb', 'ccc']),
    'empty array': np.zeros(0),
    'structured dtype': np.array([(1, 2.0)], dtype=[('a', 'i4'), ('b', 'f8')]),
    'list of arrays': [np.arange(5), np.arange(7), np.arange(3)],
    'list of lists of arrays': [[np.arange(4), np.arange(6)] for _ in range(5)],
    'nested plain lists': [[1, 2, 3], [4, 5], []],
    'tuple': (1, 'two', np.arange(3)),
    'nested tuple': ((1, 2), (3, (4, 5))),
    'bytes': b'\x00\x01\x02 binary',
    'numpy scalar': np.float32(1.5),
    'numpy int scalar': np.int16(-3),
    'dict of arrays': {'a': np.arange(10), 'b': np.zeros(3)},
    'dict with non string keys': {1: 'a', (2, 3): 'b'},
    'special floats in a list': [float('nan'), float('inf'), float('-inf'), 1.0],
    'special floats in a dict': {'x': float('nan'), 'y': float('inf')},
    'empty list': [],
    'empty dict': {},
    'none': None,
    'deeply nested': {'x': [{'y': (np.arange(2), b'z', None)}]},
}


class TestRoundTrip:

    @pytest.mark.parametrize('label', list(VALUES))
    def test_value_survives(self, label):
        value = VALUES[label]
        restored = blob_store.loads(blob_store.dumps(value))

        assert same(restored, value), f'{restored!r} != {value!r}'

    @pytest.mark.parametrize('label', list(VALUES))
    def test_value_survives_uncompressed(self, label):
        value = VALUES[label]
        restored = blob_store.loads(blob_store.dumps(value, compress=False))

        assert same(restored, value)

    def test_arrays_come_back_writable(self):
        """A view on the stored buffer would be read-only, and would keep the
        row's bytes alive for as long as the caller held the array."""
        restored = blob_store.loads(blob_store.dumps(np.arange(5)))
        restored[0] = 99

        assert restored[0] == 99

    def test_a_large_value_is_still_one_container(self):
        value = np.random.default_rng(1).random(100_000)
        restored = blob_store.loads(blob_store.dumps(value))

        assert same(restored, value)


class TestWhatTheBytesContain:

    @pytest.mark.parametrize('label', list(VALUES))
    def test_no_python_module_is_named(self, label):
        """The failure this format exists to prevent: bytes that tell the reader
        which module to import. A pickle of a list of arrays names
        numpy._core.multiarray._reconstruct, a private path numpy renamed in 2.0."""
        raw = blob_store.dumps(VALUES[label], compress=False)

        for module in (b'numpy.', b'numpy_', b'entarchy.backend', b'__main__', b'builtins'):
            assert module not in raw, f'{module!r} found in the stored bytes'

    def test_the_header_is_json(self):
        raw = blob_store.dumps({'a': np.arange(4), 'b': [1, 2]}, compress=False)
        length = int.from_bytes(raw[6:10], 'little')
        header = json.loads(raw[10:10 + length])

        assert set(header) == {'t', 'a'}
        assert header['a'] == [['<i4' if np.arange(4).dtype == np.int32 else '<i8', [4]]]

    def test_dtype_and_shape_are_readable_without_entarchy(self):
        raw = blob_store.dumps(np.zeros((3, 7), dtype=np.float32), compress=False)
        length = int.from_bytes(raw[6:10], 'little')
        header = json.loads(raw[10:10 + length])

        assert header['a'] == [['<f4', [3, 7]]]

    def test_store_of_reports_where_the_value_lives(self):
        assert blob_store.store_of(blob_store.dumps([1, 2])) == 'internal'
        assert blob_store.store_of(blob_store.dumps_external('ext/a/b.blob')) == 'ext/a/b.blob'
        assert blob_store.store_of(
            blob_store.dumps_archived('blocks/0001.asdf', 'uuid/name')) == 'asdf:blocks/0001.asdf#uuid/name'


class TestRefusesRatherThanExecutes:

    @pytest.mark.parametrize('raw', [
        b'',
        b'EN',
        b'\x80\x04\x95junk',                       # a pickle
        b'ENTB\x01\x00',                           # right marker, no header
    ], ids=['empty', 'truncated', 'a pickle', 'no header'])
    def test_malformed_input_raises(self, raw):
        with pytest.raises((ValueError, json.JSONDecodeError)):
            blob_store.loads(raw)

    def test_a_future_container_version_says_so(self):
        raw = bytearray(blob_store.dumps([1, 2]))
        raw[4] = 99

        with pytest.raises(ValueError, match='container version 99'):
            blob_store.loads(bytes(raw))

    def test_an_unknown_compression_codec_says_so(self):
        raw = bytearray(blob_store.dumps([1, 2], compress=False))
        raw[5] = 42

        with pytest.raises(ValueError, match='compression codec'):
            blob_store.loads(bytes(raw))

    def test_an_unknown_encoded_kind_says_so(self):
        with pytest.raises(ValueError, match='Unknown encoded value kind'):
            blob_store.decode({blob_store.TYPE_KEY: 'from_the_future'})

    def test_a_non_bytes_value_says_so(self):
        with pytest.raises(ValueError, match='must be bytes'):
            blob_store.loads('not bytes')


class TestCompression:

    def test_compressible_values_shrink(self):
        value = [np.arange(200) for _ in range(20)]

        assert len(blob_store.dumps(value)) < len(blob_store.dumps(value, compress=False))

    def test_incompressible_values_are_stored_raw(self):
        """Random floats do not compress, and paying zlib on every read for
        nothing is worse than the bytes saved."""
        value = np.random.default_rng(2).random(2000)

        assert len(blob_store.dumps(value)) == len(blob_store.dumps(value, compress=False))

    def test_small_values_are_not_attempted(self):
        value = [1, 2, 3]

        assert len(blob_store.dumps(value)) == len(blob_store.dumps(value, compress=False))


class TestPickleFallback:

    def test_an_unencodable_value_still_round_trips(self):
        value = np.array([{'a': 1}, {'b': 2}], dtype=object)
        restored = blob_store.loads(blob_store.dumps(value))

        assert list(restored) == list(value)

    def test_the_fallback_is_reported(self):
        report = blob_store.EncodingReport()
        blob_store.dumps(np.array([object()], dtype=object), report, where='some/attr')

        assert not report.is_fully_portable
        assert 'ndarray[object]' in report.summary()
        assert report.pickled[0][0] == 'some/attr'

    def test_ordinary_values_report_nothing(self):
        report = blob_store.EncodingReport()
        for value in VALUES.values():
            blob_store.dumps(value, report)

        assert report.is_fully_portable, report.summary()

    def test_ragged_packing_is_counted(self):
        report = blob_store.EncodingReport()
        blob_store.dumps([np.arange(3), np.arange(4)], report)

        assert report.ragged_packed == 1


class TestExternalFiles:
    """Values at or above max_blob_size go to a file, and the row keeps a pointer."""

    def test_pointer_resolves_against_the_entarchy_root(self, tmp_path):
        value = np.arange(1000)
        target = tmp_path / 'ext' / 'aa' / 'value.blob'
        target.parent.mkdir(parents=True)
        target.write_bytes(blob_store.dumps(value))

        pointer = blob_store.dumps_external('ext/aa/value.blob')

        assert same(blob_store.loads(pointer, root_path=str(tmp_path)), value)

    def test_a_pointer_without_a_root_says_so(self):
        with pytest.raises(ValueError, match='no entarchy path'):
            blob_store.loads(blob_store.dumps_external('ext/a.blob'))

    def test_the_file_path_shards_by_entity_uuid(self, tmp_path):
        entity_uuid = 'aaaabbbb-cccc-dddd-eeee-ffff00001111'
        directory, filename = _get_attribute_fp(str(tmp_path), entity_uuid, 'my/attr', 'blob')

        assert 'aaaa/bbbb/cccc' in directory
        assert filename == f'{_get_namehash("my/attr")}.blob'

    def test_two_attributes_of_one_entity_do_not_collide(self, tmp_path):
        entity_uuid = 'aaaabbbb-cccc-dddd-eeee-ffff00001111'
        first = _get_attribute_fp(str(tmp_path), entity_uuid, 'a', 'blob')
        second = _get_attribute_fp(str(tmp_path), entity_uuid, 'b', 'blob')

        assert first[0] == second[0]
        assert first[1] != second[1]
