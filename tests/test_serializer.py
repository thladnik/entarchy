"""Unit tests for the blob Serializer and its on-disk layout."""
import os
import pickle

import numpy as np
import pytest

from entarchy.backend.sqlite import Serializer, _get_attribute_fp, _get_namehash


class _FakeEntarchy:
    """Minimal stand-in - Serializer.deserialize only needs .path"""

    def __init__(self, path):
        self.path = path


def roundtrip(data, tmp_path, max_blob_size=10 * 1024 * 1024):
    ser = Serializer()
    ser.serialize(data, str(tmp_path), 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'attr', max_blob_size)
    # Storage always goes through pickle, so exercise that too
    restored: Serializer = pickle.loads(pickle.dumps(ser))
    return restored, restored.deserialize(_FakeEntarchy(str(tmp_path)))


class TestInternalStorage:

    def test_ndarray(self, tmp_path):
        data = np.random.rand(20, 4)
        ser, out = roundtrip(data, tmp_path)
        assert ser.store == 'internal'
        assert np.array_equal(data, out)
        assert out.dtype == data.dtype

    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'int8', 'int64', 'uint8', 'bool'])
    def test_ndarray_dtypes_preserved(self, tmp_path, dtype):
        data = (np.arange(10) % 2).astype(dtype)
        _, out = roundtrip(data, tmp_path)
        assert out.dtype == np.dtype(dtype)
        assert np.array_equal(data, out)

    def test_bytes_passthrough(self, tmp_path):
        data = b'\x00\x01\x02binary'
        ser, out = roundtrip(data, tmp_path)
        assert out == data

    @pytest.mark.parametrize('data', [
        None, [1, 2, 3], {'a': 1}, ('x', 'y'), {1, 2}, 'plain string',
    ])
    def test_generic_objects(self, tmp_path, data):
        _, out = roundtrip(data, tmp_path)
        assert out == data

    def test_empty_and_zero_dim_arrays(self, tmp_path):
        for data in (np.array([]), np.array(5.0), np.zeros((0, 3))):
            _, out = roundtrip(data, tmp_path)
            assert np.array_equal(data, out)
            assert out.shape == data.shape


class TestExternalStorage:

    def test_large_array_goes_to_file(self, tmp_path):
        data = np.random.rand(500)  # 4000 bytes payload
        ser, out = roundtrip(data, tmp_path, max_blob_size=1024)

        assert ser.store != 'internal'
        assert ser.store.startswith('ext/')
        assert ser.store.endswith('.npy')
        assert np.array_equal(data, out)

    def test_large_object_uses_pickle_format(self, tmp_path):
        data = ['x'] * 5000
        ser, out = roundtrip(data, tmp_path, max_blob_size=1024)
        assert ser.store.endswith('.pickle')
        assert out == data

    def test_file_layout_is_uuid_sharded(self, tmp_path):
        entity_uuid = '01234567-89ab-cdef-0123-456789abcdef'
        fp, fn = _get_attribute_fp(str(tmp_path), entity_uuid, 'my/attr', 'npy')

        shards = fp.split('ext/')[1].split('/')
        assert len(shards) == 8
        assert all(len(s) == 4 for s in shards)
        assert ''.join(shards) == entity_uuid.replace('-', '')
        assert fn == f'{_get_namehash("my/attr")}.npy'

    def test_stored_path_is_relative(self, tmp_path):
        ser = Serializer()
        ser.serialize(np.random.rand(500), str(tmp_path), 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
                      'attr', 1024)
        # A relative path keeps the entarchy directory movable
        assert not os.path.isabs(ser.store)
        assert os.path.exists(os.path.join(str(tmp_path), ser.store))

    def test_distinct_attributes_do_not_collide(self, tmp_path):
        uuid = 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
        a = Serializer(); a.serialize(np.zeros(500), str(tmp_path), uuid, 'attr_a', 1024)
        b = Serializer(); b.serialize(np.ones(500), str(tmp_path), uuid, 'attr_b', 1024)

        assert a.store != b.store
        ent = _FakeEntarchy(str(tmp_path))
        assert np.array_equal(a.deserialize(ent), np.zeros(500))
        assert np.array_equal(b.deserialize(ent), np.ones(500))

    def test_namehash_is_deterministic_and_path_safe(self):
        for name in ['a/b/c', 'plain', 's2p/attrs/x', '../weird']:
            h = _get_namehash(name)
            assert h == _get_namehash(name)
            assert h.isalnum()


class TestSizeAccounting:

    def test_internal_size_includes_payload(self, tmp_path):
        small = Serializer()
        small.serialize(np.zeros(10), str(tmp_path), 'u' * 8, 'a', 10 * 1024 * 1024)
        large = Serializer()
        large.serialize(np.zeros(10_000), str(tmp_path), 'u' * 8, 'b', 10 * 1024 * 1024)

        # Regression: __sizeof__ used to return a constant for internal blobs
        assert large.__sizeof__() > small.__sizeof__()
        assert large.__sizeof__() >= 80_000

    def test_external_size_includes_payload(self, tmp_path):
        ser = Serializer()
        ser.serialize(np.zeros(10_000), str(tmp_path), 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
                      'a', 1024)
        assert ser.__sizeof__() >= 80_000


class TestThreshold:

    def test_boundary_uses_external_storage(self, tmp_path):
        """Values at or above max_blob_size are stored externally."""
        data = np.zeros(128, dtype=np.uint8)
        ser = Serializer()
        ser.serialize(data, str(tmp_path), 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'a',
                      max_blob_size=data.nbytes)
        assert ser.store != 'internal'

    def test_just_below_boundary_stays_internal(self, tmp_path):
        data = np.zeros(128, dtype=np.uint8)
        ser = Serializer()
        ser.serialize(data, str(tmp_path), 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee', 'a',
                      max_blob_size=10 * 1024 * 1024)
        assert ser.store == 'internal'
