"""Collection writes that exceed a single SQL statement.

SQLite binds one parameter per column per row and refuses a statement with more
than SQLITE_MAX_VARIABLE_NUMBER of them, so a collection write large enough to
cross that limit used to fail with "too many SQL variables" - at roughly 2340
entities, well inside the range entarchy is built for. MySQL was unaffected
because PyMySQL interpolates literals instead of binding.
"""
import numpy as np
import pandas as pd
import pytest

import entarchy
from entarchy.backend import SQLiteBackend
from entarchy.backend.sql import MAX_BOUND_PARAMETERS, _chunk_by_bound_parameters

from conftest import Animal, DeepArchy, Layer, Recording, Roi


# Building thousands of entities is slow while entity creation costs ~10 ms each
pytestmark = pytest.mark.slow

# Three chunks at the current table width, so cross-chunk behaviour is exercised
# without the suite paying for more entities than that needs
LARGE = 4_000


@pytest.fixture(scope='module')
def wide(tmp_path_factory):
    """One layer holding more ROIs than fit in a single insert statement.

    Built once for the module: entity creation currently costs ~10 ms each, so
    rebuilding per test would dominate the suite. Tests therefore write to
    attribute names of their own and never read another test's.
    """
    base = (tmp_path_factory.mktemp('large') / 'wide').as_posix()
    ent = DeepArchy.create(base, SQLiteBackend(base, dbname='wide.db'))

    with ent:
        animal = Animal(ent, _id='animal_1', _parent=ent.root)
        ent.add_new_entity(animal)
        recording = Recording(ent, _id='rec_0', _parent=animal)
        ent.add_new_entity(recording)
        layer = Layer(ent, _id='plane0', _parent=recording)
        ent.add_new_entity(layer)

        for index in range(LARGE):
            ent.add_new_entity(Roi(ent, _id=f'roi_{index}', _parent=layer))

    yield ent
    ent.backend.close()


def _uuids(ent):
    return [uuid for uuid, _ in ent.backend.get_collection_parent_uuids(ent.get(Roi))]


class TestChunking:

    def test_rows_per_statement_respects_the_limit(self):
        records = [{'a': i} for i in range(10_000)]
        chunks = list(_chunk_by_bound_parameters(records, columns=14))

        assert sum(len(chunk) for chunk in chunks) == 10_000
        assert all(len(chunk) * 14 <= MAX_BOUND_PARAMETERS for chunk in chunks)

    def test_no_rows_yields_nothing(self):
        assert list(_chunk_by_bound_parameters([], columns=14)) == []

    def test_a_single_row_always_fits(self):
        """Even an absurd column count must not produce an empty chunk."""
        chunks = list(_chunk_by_bound_parameters([{'a': 1}], columns=10 ** 6))

        assert chunks == [[{'a': 1}]]

    def test_rows_keep_their_order(self):
        records = [{'i': i} for i in range(5_000)]
        flattened = [row for chunk in _chunk_by_bound_parameters(records, 14)
                     for row in chunk]

        assert flattened == records


class TestLargeCollectionWrites:
    """Regression: these all raised "too many SQL variables" on SQLite."""

    def test_int_attribute(self, wide):
        uuids = _uuids(wide)
        wide.get(Roi).update(pd.DataFrame({'idx_int': range(len(uuids))}, index=uuids))

        frame = wide.get(Roi).dataframe_of(['idx_int'])
        assert len(frame) == LARGE
        assert sorted(frame['idx_int'].tolist()) == list(range(LARGE))

    def test_several_attributes_at_once(self, wide):
        uuids = _uuids(wide)
        wide.get(Roi).update(pd.DataFrame({
            'multi_index': range(len(uuids)),
            'multi_score': [float(i) * 0.5 for i in range(len(uuids))],
            'multi_good': [i % 2 == 0 for i in range(len(uuids))],
            'multi_label': [f'roi_{i}' for i in range(len(uuids))],
        }, index=uuids))

        frame = wide.get(Roi).dataframe_of(
            ['multi_index', 'multi_score', 'multi_good', 'multi_label'])
        assert len(frame) == LARGE
        assert frame['multi_score'].max() == (LARGE - 1) * 0.5
        assert frame['multi_good'].sum() == LARGE // 2

    def test_blob_attribute(self, wide):
        uuids = _uuids(wide)
        wide.get(Roi).update(pd.DataFrame(
            {'blob_trace': [np.arange(4, dtype=np.float32) + i for i in range(len(uuids))]},
            index=uuids))

        roi = wide.get(Roi)[0]
        assert isinstance(roi['blob_trace'], np.ndarray)
        assert roi['blob_trace'].dtype == np.float32

    def test_special_floats_survive_chunking(self, wide):
        """The nan/inf marker columns must stay aligned with their rows."""
        uuids = _uuids(wide)
        values = [float(i) for i in range(len(uuids))]
        values[0] = float('nan')
        values[1] = float('inf')
        values[-1] = float('-inf')

        wide.get(Roi).update(pd.DataFrame({'special_value': values}, index=uuids))

        restored = wide.get(Roi).dataframe_of(['special_value'])['special_value']
        assert np.isnan(restored[uuids[0]])
        assert restored[uuids[1]] == float('inf')
        assert restored[uuids[-1]] == float('-inf')

    def test_values_land_on_the_right_entities(self, wide):
        """A chunking bug would misalign values against uuids."""
        uuids = _uuids(wide)
        expected = {uuid: index for index, uuid in enumerate(uuids)}
        wide.get(Roi).update(pd.DataFrame({'aligned': list(expected.values())},
                                          index=list(expected)))

        frame = wide.get(Roi).dataframe_of(['aligned'])
        for uuid, index in expected.items():
            assert frame.loc[uuid, 'aligned'] == index

    def test_update_overwrites_across_chunks(self, wide):
        uuids = _uuids(wide)
        wide.get(Roi).update(pd.DataFrame({'rewritten': [0] * len(uuids)}, index=uuids))
        wide.get(Roi).update(pd.DataFrame({'rewritten': range(len(uuids))}, index=uuids))

        frame = wide.get(Roi).dataframe_of(['rewritten'])
        assert sorted(frame['rewritten'].tolist()) == list(range(LARGE))

    def test_modification_times_are_set_for_every_entity(self, wide):
        uuids = _uuids(wide)
        wide.get(Roi).update(pd.DataFrame({'touched': range(len(uuids))}, index=uuids))

        # The modified-time update binds one parameter per uuid and is chunked too
        rois = wide.get(Roi)
        times = [wide.backend.get_entity_modified_time(rois[i])
                 for i in range(0, LARGE, LARGE // 10)]
        assert all(time is not None for time in times)


class TestCollectionReads:
    """Reading attributes back out of a large collection.

    `dataframe_of` used to hang here. `EntityTable.uuid.in_(query.subquery()
    .primary_key)` looks like a membership test against the collection but is
    not: primary_key is a collection of Column objects, so the subquery lands in
    the FROM clause and the comparison runs per combined row - a cartesian
    product. Small fixtures hid it; 27 000 entities against 713 000 attribute
    rows did not finish in nine minutes.
    """

    def test_uuid_subquery_is_a_select(self, wide):
        """Not a column collection, which is what caused the cross join."""
        import sqlalchemy
        from sqlalchemy.orm import Session as SASession

        from entarchy.backend.sql import _build_query_from_collection, _uuids_of

        with SASession(wide.backend.sql_engine) as session:
            query = _build_query_from_collection(wide.get(Roi), session)
            selected = _uuids_of(query)

        assert isinstance(selected, sqlalchemy.Selectable)
        assert 'SELECT' in str(selected).upper()

    def test_generated_sql_uses_a_subquery_not_a_cross_join(self, wide):
        import sqlalchemy
        from sqlalchemy.orm import Session as SASession

        from entarchy.backend.sql import (AttributeTable, EntityTable,
                                          _build_query_from_collection, _uuids_of)

        with SASession(wide.backend.sql_engine) as session:
            query = _build_query_from_collection(wide.get(Roi), session)
            statement = str(session.query(AttributeTable.name)
                            .join(EntityTable)
                            .filter(EntityTable.uuid.in_(_uuids_of(query))))

        assert 'IN (SELECT' in statement
        # The cross join rendered the subquery into FROM, next to a comma
        from_clause = statement.split('WHERE')[0]
        assert '), (SELECT' not in from_clause
        assert ', (SELECT' not in from_clause

    def test_dataframe_of_over_the_whole_collection(self, wide):
        wide.get(Roi).update(pd.DataFrame(
            {'read_index': range(LARGE), 'read_label': [f'r{i}' for i in range(LARGE)]},
            index=_uuids(wide)))

        frame = wide.get(Roi).dataframe_of(['read_index', 'read_label'])

        assert frame.shape == (LARGE, 2)
        assert sorted(frame['read_index'].tolist()) == list(range(LARGE))

    def test_attribute_names_over_the_whole_collection(self, wide):
        names = wide.backend.get_collection_attribute_names(wide.get(Roi))

        assert 'id' in names and 'uuid' in names
