"""Attribute storage: the duplicate index, the pivot's name filter, and the
query planner statistics SQLite needs.

The three came out of benchmarking EAV against JSON columns
(docs/proposals/attribute-storage.md). None changes what a query returns, so
what these tests mostly assert is exactly that.
"""
import os

import pytest
import sqlalchemy

from entarchy.backend import SQLiteBackend
from entarchy.backend.sql import AttributeTable
from entarchy.tools import optimize_storage

from conftest import DeepArchy, Layer, Recording, Roi, Session, Subject


def _url(ent):
    return f'sqlite:///{ent.path}/{ent.backend.dbname}'


def _index_names(ent, table='attributes'):
    return {index['name'] for index in
            sqlalchemy.inspect(ent.backend.sql_engine).get_indexes(table)}


def _readd_duplicate_index(ent):
    """Put the index back, as a database written by an older entarchy has it."""
    with ent.backend.sql_engine.begin() as connection:
        connection.execute(sqlalchemy.text(
            'CREATE UNIQUE INDEX ix_unique_name_per_entity_uuid '
            'ON attributes (entity_uuid, name)'))


class TestDuplicateIndexIsGone:

    def test_not_created_for_a_new_entarchy(self, ent):
        assert 'ix_unique_name_per_entity_uuid' not in _index_names(ent)

    def test_the_name_index_is_kept(self, ent):
        """It answers "which entities have this attribute", which every ../parent
        and [Ancestor] filter needs - unlike the one that was dropped."""
        assert 'ix_attributes_name' in _index_names(ent)

    def test_the_primary_key_still_enforces_uniqueness(self, populated):
        """The dropped index was unique; the primary key on the same two columns
        has to keep that promise on its own."""
        entity = populated.get(Session)[0]

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            with sqlalchemy.orm.Session(populated.backend.sql_engine) as session:
                session.add(AttributeTable(entity_uuid=entity.uuid, name='index',
                                           value_int=1, data_type='int'))
                session.commit()

    def test_upsert_still_works(self, populated):
        """The bulk write path relies on ON CONFLICT (entity_uuid, name), which
        needs a unique index over those columns to resolve against."""
        collection = populated.get(Session)
        before = sorted(int(v) for v in collection['index'])

        collection['index'] = collection['index'] + 100

        after = sorted(int(v) for v in populated.get(Session)['index'])
        assert after == [v + 100 for v in before]


class TestOptimizeStorageTool:

    def test_reports_a_current_database_as_done(self, populated):
        state = optimize_storage.inspect(_url(populated))

        assert state['has_duplicate_index'] is False
        assert state['attribute_rows'] > 0

    def test_finds_and_drops_the_duplicate_index(self, populated):
        _readd_duplicate_index(populated)
        assert optimize_storage.inspect(_url(populated))['has_duplicate_index']

        done = optimize_storage.optimize(_url(populated), apply_changes=True, verbose=False)

        assert done['dropped_index']
        assert 'ix_unique_name_per_entity_uuid' not in _index_names(populated)

    def test_dry_run_changes_nothing(self, populated):
        _readd_duplicate_index(populated)

        done = optimize_storage.optimize(_url(populated), apply_changes=False, verbose=False)

        assert not done['dropped_index']
        assert 'ix_unique_name_per_entity_uuid' in _index_names(populated)

    def test_is_repeatable(self, populated):
        _readd_duplicate_index(populated)
        optimize_storage.optimize(_url(populated), apply_changes=True, verbose=False)
        again = optimize_storage.optimize(_url(populated), apply_changes=True, verbose=False)

        assert not again['dropped_index']

    def test_collects_planner_statistics(self, populated):
        optimize_storage.optimize(_url(populated), apply_changes=True, verbose=False)

        with populated.backend.sql_engine.connect() as connection:
            rows = connection.execute(sqlalchemy.text(
                'SELECT COUNT(*) FROM sqlite_stat1')).scalar()

        assert rows > 0
        assert optimize_storage.inspect(_url(populated))['has_statistics']

    def test_refuses_something_that_is_not_an_entarchy(self, tmp_path):
        empty = (tmp_path / 'empty.db').as_posix()
        engine = sqlalchemy.create_engine(f'sqlite:///{empty}')
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text('CREATE TABLE unrelated (a INT)'))
        engine.dispose()

        with pytest.raises(SystemExit, match='attributes table'):
            optimize_storage.inspect(f'sqlite:///{empty}')


class TestPlannerStatistics:

    def test_backend_collects_them_on_close(self, tmp_path):
        base = (tmp_path / 'stats').as_posix()
        ent = DeepArchy.create(base, SQLiteBackend(base, dbname='stats.db'))
        with ent:
            from conftest import Animal
            animal = Animal(ent, _id='a', _parent=ent.root)
            ent.add_new_entity(animal)
            animal['strain'] = 'wildtype'

        # Touch the tables, since PRAGMA optimize only analyses what a session used
        len(ent.get(Animal, 'strain == "wildtype"'))
        ent.backend.close()

        engine = sqlalchemy.create_engine(f'sqlite:///{base}/stats.db')
        with engine.connect() as connection:
            has_table = connection.execute(sqlalchemy.text(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'sqlite_stat1'")).scalar()
        engine.dispose()

        assert has_table

    def test_close_survives_a_backend_that_cannot_be_analysed(self, populated, monkeypatch):
        """Statistics are an optimisation; a database that will not take them
        still has to close cleanly."""
        def explode(*args, **kwargs):
            raise RuntimeError('boom')

        monkeypatch.setattr(type(populated.backend.sql_engine), 'connect', explode)

        populated.backend.close()  # must not raise

    def test_analysis_limit_is_set_on_the_connection(self, populated):
        with populated.backend.sql_engine.connect() as connection:
            limit = connection.execute(sqlalchemy.text('PRAGMA analysis_limit')).scalar()

        assert limit == SQLiteBackend.analysis_limit


class TestPivotNameFilter:
    """get_collection_attributes restricts the join to the requested names. The
    result must be identical to what the unrestricted join produced."""

    @pytest.fixture()
    def wide(self, deep):
        """ROIs carrying far more attributes than any one read wants."""
        rois = deep.get(Roi)
        for index, roi in enumerate(rois):
            for k in range(12):
                roi[f'analysis/metric_{k}'] = float(index * 100 + k)
        return deep

    def test_reads_a_subset_correctly(self, wide):
        frame = wide.get(Roi).dataframe_of(['index', 'analysis/metric_3'])

        assert list(frame.columns) == ['index', 'analysis/metric_3']
        assert len(frame) == 12
        for _, row in frame.iterrows():
            assert row['analysis/metric_3'] % 100 == 3

    def test_reads_every_attribute_correctly(self, wide):
        collection = wide.get(Roi)
        names = [name for name in collection.columns if name != 'uuid']

        frame = collection.dataframe_of(names)

        assert set(frame.columns) == set(names)
        assert len(frame) == 12
        assert frame['index'].notna().all()

    def test_a_missing_attribute_is_still_reported(self, wide):
        with pytest.raises(AttributeError, match='not found'):
            wide.get(Roi).dataframe_of(['index', 'never_written'])

    def test_mixed_types_survive(self, deep):
        rois = deep.get(Roi)
        for roi in rois:
            roi['label'] = 'x'
            roi['flag'] = True
            roi['ratio'] = 0.5

        frame = deep.get(Roi).dataframe_of(['label', 'flag', 'ratio'])

        assert frame['label'].tolist() == ['x'] * 12
        assert frame['flag'].all()
        assert (frame['ratio'] == 0.5).all()

    def test_an_attribute_only_some_entities_have(self, deep):
        """The name filter must not turn a sparse attribute into a missing one."""
        rois = sorted(deep.get(Roi), key=lambda e: e.uuid)
        for roi in rois[:4]:
            roi['sparse'] = 1.0

        frame = deep.get(Roi).dataframe_of(['sparse'])

        assert len(frame) == 12
        assert frame['sparse'].notna().sum() == 4
