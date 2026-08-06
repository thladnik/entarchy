"""Batched commits.

Committing per entity meant one fsync each, which dominated the cost of writing
many entities. Entarchy.commit() now runs as a single transaction. That changes
observable behaviour as well as speed: a failure part way through leaves nothing
behind instead of half the entities.
"""
import pytest
import sqlalchemy

import entarchy
from entarchy.backend import SQLiteBackend
from entarchy.backend.sql import AttributeTable, EntityTable

from conftest import Session, Subject


def _row_count(ent, table):
    with sqlalchemy.orm.Session(ent.backend.sql_engine) as session:
        return session.query(table).count()


class TestBatchDepth:

    def test_batch_commits_once_on_exit(self, ent):
        backend = ent.backend
        assert backend._batch_depth == 0

        with backend.batch():
            assert backend._batch_depth == 1

        assert backend._batch_depth == 0

    def test_batches_nest(self, ent):
        backend = ent.backend

        with backend.batch():
            with backend.batch():
                assert backend._batch_depth == 2
            # The inner block must not have ended the transaction
            assert backend._batch_depth == 1

        assert backend._batch_depth == 0

    def test_depth_resets_after_an_error(self, ent):
        backend = ent.backend

        with pytest.raises(ValueError):
            with backend.batch():
                raise ValueError('boom')

        assert backend._batch_depth == 0

    def test_backend_is_usable_after_a_failed_batch(self, ent):
        """A rolled back batch must leave the session fit for the next write."""
        backend = ent.backend

        with pytest.raises(ValueError):
            with backend.batch():
                raise ValueError('boom')

        with ent:
            subject = Subject(ent, _id='afterwards', _parent=ent.root)
            ent.add_new_entity(subject)

        assert len(ent.get(Subject, 'id == "afterwards"')) == 1


class TestAllOrNothing:

    def test_a_failure_mid_commit_persists_nothing(self, ent, monkeypatch):
        """Entity rows are inserted before their attributes; both must roll back."""
        entities_before = _row_count(ent, EntityTable)

        original = type(ent.backend).set_entity_attributes
        calls = {'n': 0}

        def fail_on_third(self, _entity, names, values):
            calls['n'] += 1
            if calls['n'] == 3:
                raise RuntimeError('deliberate failure part way through')
            return original(self, _entity, names, values)

        monkeypatch.setattr(type(ent.backend), 'set_entity_attributes', fail_on_third)

        with pytest.raises(RuntimeError, match='deliberate failure'):
            with ent:
                subject = Subject(ent, _id='subject_a', _parent=ent.root)
                ent.add_new_entity(subject)
                for index in range(5):
                    ent.add_new_entity(Session(ent, _id=f'sess_{index}', _parent=subject))

        # Nothing from the failed block survived, not even the entity rows that
        #  were inserted before the attribute write failed
        assert _row_count(ent, EntityTable) == entities_before

    def test_a_successful_commit_persists_everything(self, ent):
        entities_before = _row_count(ent, EntityTable)

        with ent:
            subject = Subject(ent, _id='subject_a', _parent=ent.root)
            ent.add_new_entity(subject)
            for index in range(5):
                ent.add_new_entity(Session(ent, _id=f'sess_{index}', _parent=subject))

        assert _row_count(ent, EntityTable) == entities_before + 6
        assert len(ent.get(Session)) == 5


class TestBatchingPreservesBehaviour:

    def test_attributes_written_in_a_batch_are_readable(self, ent):
        with ent:
            subject = Subject(ent, _id='subject_a', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['strain'] = 'wildtype'
            subject['age'] = 12

        assert subject['strain'] == 'wildtype'
        assert subject['age'] == 12

    def test_later_writes_in_a_batch_see_earlier_ones(self, ent):
        """The batch flushes between statements, so reads inside it stay correct."""
        with ent:
            subject = Subject(ent, _id='subject_a', _parent=ent.root)
            ent.add_new_entity(subject)
            for index in range(3):
                ent.add_new_entity(Session(ent, _id=f'sess_{index}', _parent=subject))

        # The child rows carry a foreign key to the parent, which only resolves
        #  if the parent row was visible when they were written
        for session_entity in ent.get(Session):
            assert session_entity.parent.id == 'subject_a'

    def test_immutability_is_still_enforced(self, ent):
        with ent:
            subject = Subject(ent, _id='subject_a', _parent=ent.root)
            ent.add_new_entity(subject)

        with pytest.raises(RuntimeError, match='identity'):
            with ent:
                subject['id'] = 'renamed'

    def test_writes_outside_a_batch_still_commit(self, ent):
        """Without an entarchy context each write commits on its own."""
        subject = Subject(ent, _id='subject_a', _parent=ent.root)
        ent.add_new_entity(subject)
        subject['strain'] = 'wildtype'

        assert ent.backend._batch_depth == 0
        assert len(ent.get(Subject, 'strain == "wildtype"')) == 1

    def test_attribute_count_matches(self, ent):
        """A batch must not drop or duplicate attribute rows."""
        before = _row_count(ent, AttributeTable)

        with ent:
            subject = Subject(ent, _id='subject_a', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['strain'] = 'wildtype'

        # id and uuid are written for every new entity, plus strain
        assert _row_count(ent, AttributeTable) == before + 3
