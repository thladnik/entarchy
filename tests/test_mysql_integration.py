"""Integration tests against a real MySQL server.

Skipped unless a server is configured, so the suite still runs anywhere:

    set ENTARCHY_MYSQL_HOST=localhost
    set ENTARCHY_MYSQL_USER=<user>
    set ENTARCHY_DB_PASSWORD=<password>
    set ENTARCHY_MYSQL_SCHEMA_PREFIX=entarchy_test_   (optional)

The user needs CREATE and DROP on schemas matching the prefix. Every test creates
its own schema and drops it again.
"""
import datetime
import math
import os
import uuid

import numpy as np
import pandas as pd
import pytest

from conftest import LabArchy, Session, Subject

pymysql = pytest.importorskip('pymysql', reason='MySQL backend requires PyMySQL')

from entarchy.backend import MySQLBackend

HOST = os.environ.get('ENTARCHY_MYSQL_HOST')
USER = os.environ.get('ENTARCHY_MYSQL_USER')
PASSWORD = os.environ.get('ENTARCHY_DB_PASSWORD')
PREFIX = os.environ.get('ENTARCHY_MYSQL_SCHEMA_PREFIX', 'entarchy_test_')

pytestmark = pytest.mark.skipif(
    not (HOST and USER and PASSWORD),
    reason='no MySQL server configured (see the module docstring)')


def drop_schema(name):
    connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD)
    connection.autocommit(True)
    try:
        with connection.cursor() as cursor:
            cursor.execute(f'DROP SCHEMA IF EXISTS {name}')
    finally:
        connection.close()


@pytest.fixture()
def schema_name():
    name = f'{PREFIX}{uuid.uuid4().hex[:10]}'
    yield name
    drop_schema(name)


@pytest.fixture()
def ent(tmp_path, schema_name):
    """An entarchy backed by a throwaway MySQL schema."""
    path = (tmp_path / 'archy').as_posix()
    backend = MySQLBackend(path, dbname=schema_name, dbhost=HOST, dbuser=USER,
                           dbpassword=PASSWORD)
    entarchy = LabArchy.create(path, backend)

    yield entarchy

    entarchy.backend.close()


@pytest.fixture()
def populated(ent):
    with ent:
        subject = Subject(ent, _id='subject_a', _parent=ent.root)
        ent.add_new_entity(subject)
        subject['strain'] = 'wildtype'

        for i in range(6):
            session = Session(ent, _id=f'sess_{i}', _parent=subject)
            ent.add_new_entity(session)
            session['index'] = i
            session['score'] = float(i) * 1.5
            session['flag'] = (i % 2 == 0)

    return ent


def fresh_read(entity, key):
    entity._attribute_cache.pop(key, None)
    return entity[key]


class TestLifecycle:

    def test_create_and_reopen(self, ent, schema_name):
        assert ent.root is not None

        reopened = LabArchy(ent.path)
        assert reopened.root.uuid == ent.root.uuid
        reopened.backend.close()

    def test_schema_exists_on_the_server(self, ent, schema_name):
        connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD)
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT SCHEMA_NAME FROM information_schema.SCHEMATA '
                               'WHERE SCHEMA_NAME = %s', (schema_name,))
                assert cursor.fetchone() is not None
        finally:
            connection.close()

    def test_delete_drops_the_schema(self, tmp_path, schema_name, monkeypatch):
        import builtins
        import random

        path = (tmp_path / 'doomed').as_posix()
        backend = MySQLBackend(path, dbname=schema_name, dbhost=HOST, dbuser=USER,
                               dbpassword=PASSWORD)
        entarchy = LabArchy.create(path, backend)

        monkeypatch.setattr(random, 'choices', lambda *a, **k: list('ABC12'))
        monkeypatch.setattr(builtins, 'input', lambda *a, **k: 'ABC12')
        entarchy.delete()

        connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD)
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT SCHEMA_NAME FROM information_schema.SCHEMATA '
                               'WHERE SCHEMA_NAME = %s', (schema_name,))
                assert cursor.fetchone() is None
        finally:
            connection.close()

    def test_password_is_not_written_to_disk(self, ent):
        config = open(os.path.join(ent.path, 'entarchy.yaml')).read()
        assert PASSWORD not in config
        assert 'dbpassword' not in config


class TestScalarRoundtrip:

    @pytest.mark.parametrize('value', [
        42, -7, 3.5, 'text', '', True, False,
        datetime.date(2024, 1, 31), datetime.datetime(2024, 1, 31, 12, 30, 5),
    ])
    def test_native_scalars(self, populated, value):
        entity = populated.get(Session)[0]
        entity['attr'] = value

        result = fresh_read(entity, 'attr')
        assert result == value
        assert type(result) is type(value)

    def test_nan_and_infinities(self, populated):
        """MySQL rejects nan and inf in a DOUBLE column, so they are stored as
        NULL plus marker flags, with the sign of inf in value_int."""
        entity = populated.get(Session)[0]

        entity['v_nan'] = float('nan')
        assert math.isnan(fresh_read(entity, 'v_nan'))

        entity['v_pos'] = float('inf')
        assert fresh_read(entity, 'v_pos') == float('inf')

        entity['v_neg'] = float('-inf')
        assert fresh_read(entity, 'v_neg') == float('-inf')

    def test_numpy_scalars_become_native(self, populated):
        entity = populated.get(Session)[0]
        entity['np_float'] = np.float32(1.5)
        entity['np_int'] = np.int64(7)

        assert fresh_read(entity, 'np_float') == 1.5
        assert fresh_read(entity, 'np_int') == 7


class TestBlobs:

    def test_small_array(self, populated):
        entity = populated.get(Session)[0]
        arr = np.random.rand(50, 3)
        entity['arr'] = arr

        assert np.array_equal(fresh_read(entity, 'arr'), arr)

    def test_blob_larger_than_64k(self, populated):
        """The generic BLOB type caps at 64 kB on MySQL, which would silently
        truncate stored arrays; the column is declared LONGBLOB for that dialect."""
        entity = populated.get(Session)[0]
        arr = np.random.rand(40_000)  # 320 kB payload
        assert arr.nbytes > 64 * 1024

        entity['big'] = arr

        assert np.array_equal(fresh_read(entity, 'big'), arr)

    def test_external_storage(self, populated):
        populated.max_blob_size = 1024
        entity = populated.get(Session)[0]
        arr = np.random.rand(5000)

        entity['external'] = arr

        assert np.array_equal(fresh_read(entity, 'external'), arr)
        assert os.path.isdir(os.path.join(populated.path, 'ext'))

    def test_generic_objects(self, populated):
        entity = populated.get(Session)[0]
        for key, value in [('list', [1, 2, 'x']), ('dict', {'a': [1, 2]}), ('none', None)]:
            entity[key] = value
            assert fresh_read(entity, key) == value


class TestCollectionWrites:
    """The upsert path, which uses ON DUPLICATE KEY UPDATE on this dialect."""

    def test_scalar_broadcast(self, populated):
        sessions = populated.get(Session)
        sessions['label'] = 'x'

        assert list(populated.get(Session)['label']) == ['x'] * 6

    def test_float_series(self, populated):
        sessions = populated.get(Session)
        values = pd.Series({u: float(i) for i, u in enumerate(sessions.index)})
        sessions['c_float'] = values

        out = populated.get(Session)['c_float']
        for u in sessions.index:
            assert out.loc[u] == values.loc[u]

    def test_float_special_values(self, populated):
        sessions = populated.get(Session)
        uuids = list(sessions.index)
        values = pd.Series(index=uuids, dtype='float64',
                           data=[1.5, float('nan'), float('inf'), float('-inf'), 0.0, -2.5])

        sessions['c_special'] = values

        out = populated.get(Session)['c_special']
        assert out.loc[uuids[0]] == 1.5
        assert np.isnan(out.loc[uuids[1]])
        assert out.loc[uuids[2]] == np.inf
        assert out.loc[uuids[3]] == -np.inf

    def test_array_blobs_roundtrip(self, populated):
        """Regression: the collection path used to store a pickled pandas Series."""
        sessions = populated.get(Session)
        arrays = {u: np.arange(4.0) + i for i, u in enumerate(sessions.index)}

        sessions['c_arr'] = pd.Series(arrays)

        out = populated.get(Session)['c_arr']
        for u, arr in arrays.items():
            assert np.array_equal(out.loc[u], arr)

    def test_upsert_overwrites_and_changes_type(self, populated):
        sessions = populated.get(Session)
        sessions['morph'] = 1
        sessions['morph'] = 'now_a_string'

        assert list(populated.get(Session)['morph']) == ['now_a_string'] * 6

    def test_dates_through_the_collection(self, populated):
        sessions = populated.get(Session)
        day = datetime.date(2024, 3, 1)
        moment = datetime.datetime(2024, 3, 1, 8, 15, 0)

        sessions['when'] = pd.Series({u: day for u in sessions.index})
        sessions['stamp'] = pd.Series({u: moment for u in sessions.index})

        out = populated.get(Session).dataframe_of(['when', 'stamp'])
        assert pd.Timestamp(out['when'].iloc[0]).date() == day
        assert pd.Timestamp(out['stamp'].iloc[0]).to_pydatetime() == moment


class TestQueries:

    def indices(self, collection):
        return sorted(collection['index'].tolist())

    def test_comparisons(self, populated):
        assert self.indices(populated.get(Session, 'index > 3')) == [4, 5]
        assert self.indices(populated.get(Session, 'index != 3')) == [0, 1, 2, 4, 5]
        assert self.indices(populated.get(Session, 'flag == True')) == [0, 2, 4]

    def test_in_operator(self, populated):
        assert self.indices(populated.get(Session, 'index IN (1, 3, 5)')) == [1, 3, 5]

    def test_precedence(self, populated):
        result = self.indices(populated.get(Session, 'index <= 1 AND flag == True OR index >= 4'))
        assert result == [0, 4, 5]

    def test_xor(self, populated):
        assert self.indices(populated.get(Session, 'index <= 2 XOR flag == True')) == [1, 4]

    def test_exist(self, populated):
        first = populated.get(Session)[0]
        first['only_here'] = 1

        assert len(populated.get(Session, 'EXIST(only_here)')) == 1
        assert len(populated.get(Session, 'NOT(EXIST(only_here))')) == 5

    def test_parent_traversal(self, populated):
        assert len(populated.get(Session, '[Subject]strain == "wildtype"')) == 6
        assert len(populated.get(Session, '../strain == "wildtype"')) == 6

    def test_slicing(self, populated):
        sessions = populated.get(Session)
        forward = [e.uuid for e in sessions[:]]

        assert [e.uuid for e in sessions[::-1]] == forward[::-1]
        assert len(sessions[::2]) == 3

    def test_dataframe_with_parent_attributes(self, populated):
        df = populated.get(Session).dataframe_of(['index', '[Subject]strain'])

        assert len(df) == 6
        assert set(df['[Subject]strain']) == {'wildtype'}


class TestModificationTracking:

    def test_triggers_flag_matches_the_server(self, ent):
        """Trigger creation needs privileges the user may not have; either way the
        flag must reflect reality, because it decides whether modification times
        are maintained in Python."""
        connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD)
        try:
            with connection.cursor() as cursor:
                cursor.execute('SELECT COUNT(*) FROM information_schema.TRIGGERS '
                               'WHERE TRIGGER_SCHEMA = %s', (ent.backend.dbname,))
                trigger_count = cursor.fetchone()[0]
        finally:
            connection.close()

        assert ent.backend.db_triggers_enabled == (trigger_count > 0)

    def test_modified_time_advances_on_write(self, populated):
        entity = populated.get(Session)[0]
        before = populated.backend.get_entity_modified_time(entity)

        entity['something_new'] = 1

        assert populated.backend.get_entity_modified_time(entity) >= before


class TestDigestMode:

    def test_ingested_attributes_become_immutable(self, ent):
        import entarchy

        @entarchy.digest_method
        def ingest(entarchy_obj):
            with entarchy_obj:
                subject = Subject(entarchy_obj, _id='ingested', _parent=entarchy_obj.root)
                entarchy_obj.add_new_entity(subject)
                subject['raw'] = 1
            return subject

        subject = ingest(ent)

        with pytest.raises(RuntimeError, match='immutable'):
            subject['raw'] = 2


class TestDatetimePrecisionMigration:
    """Entarchies created before the fix have whole-second timestamp columns."""

    @staticmethod
    def downgrade(schema_name):
        """Recreate the old column definitions, to stand in for an existing database."""
        connection = pymysql.connect(host=HOST, user=USER, password=PASSWORD, database=schema_name)
        connection.autocommit(True)
        try:
            with connection.cursor() as cursor:
                for table, column, null in [('entities', 'created', 'NOT NULL'),
                                            ('entities', 'modified', 'NOT NULL'),
                                            ('attributes', 'created', 'NOT NULL'),
                                            ('attributes', 'modified', 'NOT NULL'),
                                            ('attributes', 'value_datetime', 'NULL')]:
                    cursor.execute(f'ALTER TABLE {table} MODIFY {column} DATETIME {null}')
        finally:
            connection.close()

    def url_for(self, schema_name):
        from urllib.parse import quote_plus
        return (f'mysql+pymysql://{quote_plus(USER)}:{quote_plus(PASSWORD)}'
                f'@{HOST}/{schema_name}')

    def test_new_entarchies_already_have_precision(self, ent, schema_name):
        from entarchy.tools import migrate_datetime_precision

        findings = migrate_datetime_precision.inspect(self.url_for(schema_name))

        assert findings
        assert not any(f['needs_migration'] for f in findings)

    def test_detects_and_fixes_old_columns(self, ent, schema_name):
        from entarchy.tools import migrate_datetime_precision

        self.downgrade(schema_name)
        url = self.url_for(schema_name)

        findings = migrate_datetime_precision.inspect(url)
        assert sum(f['needs_migration'] for f in findings) == 5

        # Dry run changes nothing
        migrate_datetime_precision.migrate(url, apply_changes=False)
        assert sum(f['needs_migration'] for f in migrate_datetime_precision.inspect(url)) == 5

        assert migrate_datetime_precision.migrate(url, apply_changes=True) == 5
        assert not any(f['needs_migration'] for f in migrate_datetime_precision.inspect(url))

    def test_migration_restores_visibility_of_new_entities(self, ent, schema_name):
        """The defect the migration exists for: entities written moments ago are
        rounded into the future and disappear from collections."""
        from entarchy.tools import migrate_datetime_precision

        self.downgrade(schema_name)

        reopened = LabArchy(ent.path)
        try:
            missing = 0
            for attempt in range(12):
                with reopened:
                    subject = Subject(reopened, _id=f'subject_{attempt}', _parent=reopened.root)
                    reopened.add_new_entity(subject)

                if len(reopened.get(Subject, f'id == "subject_{attempt}"')) == 0:
                    missing += 1

            assert missing > 0, 'expected the rounding defect to hide at least one entity'
        finally:
            reopened.backend.close()

        migrate_datetime_precision.migrate(self.url_for(schema_name), apply_changes=True)

        reopened = LabArchy(ent.path)
        try:
            for attempt in range(12):
                with reopened:
                    subject = Subject(reopened, _id=f'after_{attempt}', _parent=reopened.root)
                    reopened.add_new_entity(subject)

                assert len(reopened.get(Subject, f'id == "after_{attempt}"')) == 1
        finally:
            reopened.backend.close()

    def test_sqlite_is_a_no_op(self, tmp_path, capsys):
        from entarchy.tools import migrate_datetime_precision

        assert migrate_datetime_precision.migrate(f'sqlite:///{(tmp_path / "x.db").as_posix()}') == 0
        assert 'full precision already' in capsys.readouterr().out


@pytest.mark.slow
class TestParallel:

    def test_map_async_against_mysql(self, populated):
        """Each worker opens its own connection to the server."""
        import _mp_worker
        from entarchy.core.entity import shutdown_worker_pool

        try:
            populated.get(Session).map_async(_mp_worker.double_score, _worker_num=2,
                                             _calibrate=False)
        finally:
            shutdown_worker_pool()

        out = populated.get(Session)[['score', 'doubled']]
        assert (out['doubled'] == out['score'] * 2).all()
