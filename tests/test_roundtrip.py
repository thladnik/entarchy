import datetime
import math
import os

import numpy as np
import pandas as pd
import pytest

from conftest import Session, Subject


def fresh_read(entity, key):
    """Read attribute with a guaranteed database round-trip (bypass cache)."""
    entity._attribute_cache.pop(key, None)
    return entity[key]


@pytest.fixture()
def subject(populated):
    return populated.get(Subject)[0]


class TestEntityScalarRoundtrip:

    @pytest.mark.parametrize('value', [
        42, -7, 3.5, 'text', '', True, False,
        datetime.date(2024, 1, 31), datetime.datetime(2024, 1, 31, 12, 30, 5),
    ])
    def test_native_scalars(self, subject, value):
        subject['attr'] = value
        result = fresh_read(subject, 'attr')
        assert result == value
        assert type(result) is type(value)

    def test_numpy_scalars_become_native(self, subject):
        subject['np_float'] = np.float32(1.5)
        subject['np_int'] = np.int64(7)
        assert fresh_read(subject, 'np_float') == 1.5
        assert type(fresh_read(subject, 'np_float')) is float
        assert fresh_read(subject, 'np_int') == 7
        assert type(fresh_read(subject, 'np_int')) is int

    def test_nan(self, subject):
        subject['v'] = float('nan')
        assert math.isnan(fresh_read(subject, 'v'))

    def test_positive_inf(self, subject):
        subject['v'] = float('inf')
        assert fresh_read(subject, 'v') == float('inf')

    def test_negative_inf(self, subject):
        subject['v'] = float('-inf')
        assert fresh_read(subject, 'v') == float('-inf')

    def test_none_and_containers_roundtrip_as_blob(self, subject):
        for key, value in [('none', None), ('list', [1, 2, 'x']), ('dict', {'a': 1, 'b': [2, 3]})]:
            subject[key] = value
            assert fresh_read(subject, key) == value


class TestEntityBlobRoundtrip:

    def test_array_internal(self, subject):
        arr = np.random.rand(50, 3)
        subject['arr'] = arr
        out = fresh_read(subject, 'arr')
        assert np.array_equal(arr, out)

    def test_array_external(self, populated, subject):
        populated.max_blob_size = 256  # force external file storage
        arr = np.random.rand(100, 10)
        subject['big'] = arr
        assert np.array_equal(arr, fresh_read(subject, 'big'))
        # An external file must exist and the DB row must not embed the payload
        ext_dir = os.path.join(populated.path, 'ext')
        assert os.path.isdir(ext_dir)

    def test_data_size_is_what_was_stored(self, populated, subject):
        """Bytes actually written, not the size of the value in memory - an
        array of zeros compresses to a fraction of its 8000 payload bytes."""
        import sqlite3

        arr = np.zeros(1000, dtype=np.float64)
        subject['sized'] = arr

        con = sqlite3.connect(os.path.join(populated.path, 'test.db'))
        size, stored = con.execute(
            "SELECT data_size, LENGTH(value_blob) FROM attributes "
            "WHERE name = 'sized'").fetchone()
        con.close()

        assert size == stored

    def test_data_size_tracks_the_value_for_incompressible_data(self, populated, subject):
        arr = np.random.default_rng(0).random(1000)
        subject['noise'] = arr

        import sqlite3
        con = sqlite3.connect(os.path.join(populated.path, 'test.db'))
        size = con.execute("SELECT data_size FROM attributes WHERE name = 'noise'").fetchone()[0]
        con.close()

        assert size >= arr.nbytes


class TestCollectionWriteRoundtrip:
    """Writes through the collection path (set_collection_attributes)."""

    def test_scalar_types(self, populated):
        sessions = populated.get(Session)
        sessions['c_int'] = 5
        sessions['c_str'] = 'hello'
        sessions['c_bool'] = True

        fresh = populated.get(Session)
        assert list(fresh['c_int']) == [5] * len(fresh)
        assert list(fresh['c_str']) == ['hello'] * len(fresh)
        assert list(fresh['c_bool']) == [True] * len(fresh)

    def test_float_series(self, populated):
        sessions = populated.get(Session)
        values = pd.Series({u: float(i) for i, u in enumerate(sessions.index)})
        sessions['c_float'] = values

        fresh = populated.get(Session)
        out = fresh['c_float']
        for u in sessions.index:
            assert out.loc[u] == values.loc[u]

    def test_float_special_values(self, populated):
        sessions = populated.get(Session)
        uuids = list(sessions.index)
        values = pd.Series(index=uuids, dtype='float64',
                           data=[1.5, float('nan'), float('inf'), float('-inf'), 0.0, -2.5])
        sessions['c_special'] = values

        # Collection read
        out = populated.get(Session)['c_special']
        assert out.loc[uuids[0]] == 1.5
        assert np.isnan(out.loc[uuids[1]])
        assert out.loc[uuids[2]] == np.inf
        assert out.loc[uuids[3]] == -np.inf
        assert out.loc[uuids[4]] == 0.0
        assert out.loc[uuids[5]] == -2.5

        # Single-entity read of the same rows
        e_nan = populated.get_entity_by_uuid(uuids[1])
        assert math.isnan(fresh_read(e_nan, 'c_special'))
        e_neginf = populated.get_entity_by_uuid(uuids[3])
        assert fresh_read(e_neginf, 'c_special') == float('-inf')

    def test_array_blobs_readable_again(self, populated):
        """Regression: collection-written blobs used to store a pickled pandas
        Series and were unreadable afterwards."""
        sessions = populated.get(Session)
        arrays = {u: np.arange(4.0) + i for i, u in enumerate(sessions.index)}
        sessions['c_arr'] = pd.Series(arrays)

        # Read via a fresh collection
        out = populated.get(Session)['c_arr']
        for u, arr in arrays.items():
            assert np.array_equal(out.loc[u], arr)

        # Read via single entity
        entity = populated.get_entity_by_uuid(list(arrays)[0])
        assert np.array_equal(fresh_read(entity, 'c_arr'), arrays[list(arrays)[0]])

    def test_array_blobs_external(self, populated):
        populated.max_blob_size = 128
        sessions = populated.get(Session)
        arrays = {u: np.random.rand(64) for u in sessions.index}
        sessions['c_big'] = pd.Series(arrays)

        out = populated.get(Session)['c_big']
        for u, arr in arrays.items():
            assert np.array_equal(out.loc[u], arr)

        # Payload must live on disk, not embedded in the database rows
        import sqlite3
        con = sqlite3.connect(os.path.join(populated.path, 'test.db'))
        max_blob = con.execute(
            "SELECT MAX(LENGTH(value_blob)) FROM attributes WHERE name = 'c_big'").fetchone()[0]
        con.close()
        assert max_blob < 1024  # only the pointer to the file, not the payload

    def test_update_overwrites_previous_type(self, populated):
        sessions = populated.get(Session)
        sessions['c_morph'] = 1
        sessions['c_morph'] = 'now_a_string'
        out = populated.get(Session)['c_morph']
        assert list(out) == ['now_a_string'] * len(sessions)
