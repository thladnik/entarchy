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


class TestAttributeStorage:
    """Attribute storage against a real server, where the dialect differs.

    InnoDB backs the primary key with the clustered index rather than a separate
    one, and column type limits are enforced here where SQLite ignores them.
    """

    def url_for(self, schema_name):
        from urllib.parse import quote_plus
        return (f'mysql+pymysql://{quote_plus(USER)}:{quote_plus(PASSWORD)}'
                f'@{HOST}/{schema_name}')

    def test_the_server_keeps_its_own_statistics(self, populated, schema_name):
        from entarchy.tools import optimize_storage

        state = optimize_storage.inspect(self.url_for(schema_name))

        assert state['dialect'] == 'mysql'
        assert state['has_statistics'] is None
        assert state['needs_work'] is False

    def test_the_primary_key_alone_enforces_uniqueness(self, populated, schema_name):
        """There is no separate unique index on (entity_uuid, name) - InnoDB's
        clustered index has to carry it."""
        import sqlalchemy
        import sqlalchemy.orm

        from entarchy.backend.sql import AttributeTable

        entity = populated.get(Session)[0]

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            with sqlalchemy.orm.Session(populated.backend.sql_engine) as session:
                session.add(AttributeTable(entity_uuid=entity.uuid, name='index',
                                           value_int=1, data_type='int'))
                session.commit()

    def test_long_strings_round_trip(self, populated):
        """The reason value_str is Text: as String(500) this raised
        DataError 1406 on MySQL while working fine on SQLite."""
        entity = populated.get(Session)[0]

        for length in (499, 500, 501, 5000, 100_000):
            text = 'x' * length
            entity['note'] = text
            assert fresh_read(entity, 'note') == text

    def test_a_long_string_is_still_filterable(self, populated):
        entity = populated.get(Session)[0]
        entity['note'] = 'y' * 2000

        assert len(populated.get(Session, f'note == "{"y" * 2000}"')) == 1

    def test_blobs_round_trip_through_the_server(self, populated):
        """LONGBLOB carries the container unchanged - no encoding surprises on
        the way through PyMySQL."""
        import numpy as np

        entity = populated.get(Session)[0]
        values = {
            'array': np.arange(500, dtype=np.float32),
            'ragged': [np.arange(3), np.arange(5)],
            'mapping': {'a': np.zeros(4), 'b': 'text'},
            'raw': b'\x00\x01\x02binary',
        }
        for name, value in values.items():
            entity[name] = value

        for name, value in values.items():
            got = fresh_read(entity, name)
            if isinstance(value, np.ndarray):
                assert np.array_equal(got, value) and got.dtype == value.dtype
            elif isinstance(value, list):
                assert all(np.array_equal(a, b) for a, b in zip(got, value))
            elif isinstance(value, dict):
                assert np.array_equal(got['a'], value['a']) and got['b'] == value['b']
            else:
                assert got == value

    def test_collection_read_still_includes_entities_without_the_attribute(self, ent):
        """The pivot's name restriction must not drop them (it is an outer join)."""
        with ent:
            subject = Subject(ent, _id='subject', _parent=ent.root)
            ent.add_new_entity(subject)
            for i in range(6):
                session_entity = Session(ent, _id=f's_{i}', _parent=subject)
                ent.add_new_entity(session_entity)
                session_entity['index'] = i
                if i < 2:
                    session_entity['sparse'] = float(i)

        frame = ent.get(Session).dataframe_of(['index', 'sparse'])

        assert len(frame) == 6
        assert frame['sparse'].notna().sum() == 2


class TestLinkSchema:
    """The link tables against a real server.

    Worth its own coverage because MySQL rejects things SQLite accepts: index
    key lengths (utf8mb4 counts four bytes per character, against a 3072 byte
    InnoDB limit), and self-referential foreign keys of the kind link_types
    carries for link endpoints.
    """

    def test_tables_are_created(self, ent):
        from sqlalchemy import inspect

        tables = set(inspect(ent.backend.sql_engine).get_table_names())

        assert 'links' in tables
        assert 'link_types' in tables

    def test_indexes_fit_the_key_length_limit(self, ent):
        """The composite unique index is the one at risk."""
        from sqlalchemy import inspect

        indexes = inspect(ent.backend.sql_engine).get_indexes('links')
        names = {index['name'] for index in indexes}

        assert 'ix_unique_link_per_type_and_pair' in names
        assert 'ix_link_reverse' in names

    def test_define_and_read_back(self, ent):
        ent.define_link_type('mean_response', Subject, Session,
                             description='trial-averaged response')

        spec = ent.get_link_type('mean_response')
        assert spec.linker.entity_type == 'Subject'
        assert spec.linked.entity_type == 'Session'
        assert spec.description == 'trial-averaged response'

    def test_self_referential_endpoint(self, ent):
        """link_types.linker_link_type points back at link_types.name."""
        ent.define_link_type('mean_response', Subject, Session)
        ent.define_link_type('adaptation', 'mean_response', 'mean_response',
                             symmetric=False)

        spec = ent.get_link_type('adaptation')
        assert spec.linker.link_type == 'mean_response'

    def test_symmetric_and_cardinality_persist(self, ent):
        ent.define_link_type('correlated', Session, Session, symmetric=True,
                             cardinality='one_per_linker')

        spec = ent.get_link_type('correlated')
        assert spec.symmetric is True
        assert spec.cardinality == 'one_per_linker'

    def test_a_pair_may_carry_several_kinds(self, populated):
        from conftest import make_link_row

        populated.define_link_type('mean_response', Subject, Session)
        populated.define_link_type('peak_latency', Subject, Session)

        subject = populated.get(Subject)[0]
        session_entity = populated.get(Session)[0]
        make_link_row(populated, 'mean_response', subject.uuid, session_entity.uuid)
        make_link_row(populated, 'peak_latency', subject.uuid, session_entity.uuid)

        assert populated.backend.count_links_of_type('mean_response') == 1
        assert populated.backend.count_links_of_type('peak_latency') == 1

    def test_removing_links_removes_their_carriers(self, populated):
        from conftest import make_link_row

        populated.define_link_type('mean_response', Subject, Session)
        subject = populated.get(Subject)[0]
        for session_entity in populated.get(Session):
            make_link_row(populated, 'mean_response', subject.uuid, session_entity.uuid)

        removed = populated.backend.remove_links_of_type('mean_response')

        assert removed == 6
        assert populated.backend.count_links_of_type('mean_response') == 0


class TestLinkWrites:
    """Creating links against a real server.

    The bulk path inserts entity and link rows through the core rather than the
    ORM, so the statements it builds are dialect specific in a way the ORM
    normally hides.
    """

    def test_single_link_round_trip(self, populated):
        subject = populated.get(Subject)[0]
        session_entity = populated.get(Session)[0]

        with populated:
            link = populated.link(subject, session_entity, 'mean_response',
                                  mean_dff=0.42)

        assert link.link_type == 'mean_response'
        assert fresh_read(link, 'mean_dff') == 0.42
        assert populated.get_link(subject, session_entity,
                                  'mean_response').uuid == link.uuid

    def test_bulk_write(self, populated):
        subject = populated.get(Subject)[0]
        sessions = list(populated.get(Session))

        result = populated.link_from_frame(pd.DataFrame({
            'linker_uuid': [subject.uuid] * len(sessions),
            'linked_uuid': [s.uuid for s in sessions],
            'mean_dff': [float(i) for i in range(len(sessions))],
            'label': [f'row_{i}' for i in range(len(sessions))],
        }), 'mean_response')

        assert result.created == len(sessions)
        assert populated.backend.count_links_of_type('mean_response') == len(sessions)

        first = populated.get_link(subject, sessions[0], 'mean_response')
        assert fresh_read(first, 'label') == 'row_0'

    def test_bulk_write_is_idempotent(self, populated):
        subject = populated.get(Subject)[0]
        sessions = list(populated.get(Session))
        frame = pd.DataFrame({'linker_uuid': [subject.uuid] * len(sessions),
                              'linked_uuid': [s.uuid for s in sessions]})

        populated.link_from_frame(frame, 'mean_response')
        again = populated.link_from_frame(frame, 'mean_response')

        assert again.created == 0
        assert again.already_present == len(sessions)

    def test_symmetric_links_store_once(self, populated):
        sessions = list(populated.get(Session))
        populated.define_link_type('correlated', Session, Session, symmetric=True)

        with populated:
            populated.link(sessions[0], sessions[1], 'correlated', r=0.9)
            populated.link(sessions[1], sessions[0], 'correlated')

        assert populated.backend.count_links_of_type('correlated') == 1
        assert populated.get_link(sessions[1], sessions[0], 'correlated')['r'] == 0.9

    def test_counts_per_kind(self, populated):
        """Grouped and counted in the database, which is where the dialects can
        differ - the entity repr calls this on every display."""
        subject = populated.get(Subject)[0]
        sessions = list(populated.get(Session))

        with populated:
            for session_entity in sessions:
                populated.link(subject, session_entity, 'mean_response')
            populated.link(subject, sessions[0], 'peak_latency')

        assert subject.link_counts() == {'mean_response': len(sessions),
                                         'peak_latency': 1}
        assert sessions[0].link_counts() == {'mean_response': 1, 'peak_latency': 1}
        assert sessions[1].link_counts() == {'mean_response': 1}

    def test_blob_attribute_on_a_link(self, populated):
        """Links get the full attribute machinery, LONGBLOB included."""
        subject = populated.get(Subject)[0]
        session_entity = populated.get(Session)[0]

        with populated:
            link = populated.link(subject, session_entity, 'mean_response',
                                  trace=np.arange(1000, dtype=np.float32))

        restored = fresh_read(link, 'trace')
        assert isinstance(restored, np.ndarray)
        assert restored.dtype == np.float32
        assert len(restored) == 1000


class TestLinkQueries:
    """Endpoint filters build aliased joins and OR-ed subqueries, so they are
    worth running against a real server rather than only compiling."""

    @pytest.fixture()
    def responses(self, populated):
        subject = populated.get(Subject)[0]
        sessions = sorted(populated.get(Session), key=lambda e: e['index'])

        populated.define_link_type('mean_response', Subject, Session,
                                   cardinality='dense')
        populated.link_from_frame(pd.DataFrame({
            'linker_uuid': [subject.uuid] * len(sessions),
            'linked_uuid': [s.uuid for s in sessions],
            'mean_dff': [0.1 * (index + 1) for index in range(len(sessions))],
        }), 'mean_response')

        return populated, subject, sessions

    def test_own_attribute_filter(self, responses):
        ent, subject, sessions = responses

        assert len(ent.links('mean_response', 'mean_dff > 0.35')) == 3

    def test_endpoint_by_type(self, responses):
        ent, subject, sessions = responses

        assert len(ent.links('mean_response', '@Session.flag == True')) == 3
        assert len(ent.links('mean_response', '@Subject.strain == "wildtype"')) == 6

    def test_endpoint_by_role(self, responses):
        ent, subject, sessions = responses

        assert len(ent.links('mean_response', '@linked.index IN (0, 1)')) == 2
        assert len(ent.links('mean_response', '@linker.strain == "wildtype"')) == 6

    def test_either_and_both(self, responses):
        ent, subject, sessions = responses

        assert len(ent.links('mean_response', '@either.index == 0')) == 1
        assert len(ent.links('mean_response', '@both.index == 0')) == 0

    def test_ancestor_traversal(self, responses):
        ent, subject, sessions = responses

        assert len(ent.links('mean_response', '@Session.[Subject]strain == "wildtype"')) == 6

    def test_combined_filters(self, responses):
        ent, subject, sessions = responses

        selected = ent.links('mean_response',
                             '@Session.flag == True AND mean_dff > 0.35')

        # flag is True at indices 0, 2 and 4, whose mean_dff are 0.1, 0.3 and 0.5
        assert len(selected) == 1
        assert selected[0]['mean_dff'] == pytest.approx(0.5)

    def test_symmetric_kind(self, populated):
        sessions = sorted(populated.get(Session), key=lambda e: e['index'])
        populated.define_link_type('correlated', Session, Session, symmetric=True)

        with populated:
            populated.link(sessions[1], sessions[0], 'correlated', r=0.9)

        assert len(populated.links('correlated', '@Session.index == 0')) == 1
        with pytest.raises(ValueError, match='symmetric'):
            len(populated.links('correlated', '@linker.index == 0'))


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
