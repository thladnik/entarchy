import sys

import pytest

from conftest import Animal, Layer, Recording, Roi, Session, Subject
from entarchy.core import console


class TestGet:
    """entity.get(Type, ...) is ent.get(Type, ...) within the entity's subtree."""

    def test_children(self, populated):
        subject = populated.get(Subject)[0]
        sessions = subject.get(Session)

        assert sorted(s.id for s in sessions) == [f'sess_{i}' for i in range(6)]
        assert all(s.parent.uuid == subject.uuid for s in sessions)

    def test_filters_and_equalities(self, populated):
        subject = populated.get(Subject)[0]

        assert sorted(s['index'] for s in subject.get(Session, 'index > 2')) == [3, 4, 5]
        assert sorted(s['index'] for s in subject.get(Session, 'index > 2', flag=True)) == [4]
        assert len(subject.get('Session', score=0.0)) == 1

    def test_any_depth_down_and_only_this_subtree(self, deep):
        animal = deep.get(Animal)[0]
        rec_0, rec_1 = sorted(animal.get(Recording), key=lambda r: r.id)

        assert len(animal.get(Roi)) == 12
        assert len(rec_0.get(Roi)) == 6
        assert len(rec_0.get(Layer, depth=0.0)) == 1
        assert {r.parent.parent.uuid for r in rec_1.get(Roi)} == {rec_1.uuid}

    def test_the_typed_collection_comes_back(self, deep):
        rois = deep.get(Animal)[0].get(Roi)

        assert type(rois) is type(deep.get(Roi))
        assert rois.sort('index').dataframe_of(['index']).shape == (12, 1)

    def test_the_root_entity_is_the_whole_store(self, populated):
        assert len(populated.root.get(Session, 'index > 2')) == 3


class TestIdentityMap:

    def test_same_uuid_returns_same_object(self, populated):
        first = populated.get(Session)[0]
        again = Session(_entarchy=populated, _uuid=first.uuid, _id=first.id)
        assert again is first

    def test_reconstruction_preserves_pending_updates(self, populated):
        """Regression: re-running __init__ on an identity-mapped instance used to
        reset the attribute cache and silently drop pending updates."""
        with populated:
            entity = populated.get(Session)[0]
            entity['pending'] = 123
            assert 'pending' in entity._attributes_to_update

            # Fetching the same entity again (e.g. via a collection) re-enters
            # __init__ with a stale init cache
            same = Session(_entarchy=populated, _uuid=entity.uuid, _id=entity.id,
                           _init_cache={'pending': 999, 'other': 5})

            assert same is entity
            assert entity._attribute_cache['pending'] == 123  # not overwritten by stale cache
            assert entity._attribute_cache['other'] == 5      # new keys are merged
            assert 'pending' in entity._attributes_to_update  # update not dropped

        # After the context commits, the value must be the pending one
        entity._attribute_cache.clear()
        assert entity['pending'] == 123


class TestAttributeSemantics:

    def test_contains(self, populated):
        entity = populated.get(Session)[0]
        assert 'score' in entity
        assert 'nope' not in entity

    def test_update_dict(self, populated):
        entity = populated.get(Session)[0]
        entity.update({'a': 1, 'b': 'two'})
        entity._attribute_cache.clear()
        assert entity['a'] == 1
        assert entity['b'] == 'two'

    def test_multi_key_get(self, populated):
        entity = populated.get(Session)[0]
        index, score = entity[['index', 'score']]
        assert score == index * 1.5

    def test_to_dict(self, populated):
        entity = populated.get(Session)[0]
        d = entity.to_dict()
        assert d['index'] == entity['index']
        assert 'score' in d

    def test_id_and_uuid_immutable(self, populated):
        entity = populated.get(Session)[0]
        with pytest.raises(RuntimeError):
            entity['id'] = 'renamed'
        with pytest.raises(RuntimeError):
            entity['uuid'] = 'clobbered'

    def test_path(self, populated):
        entity = populated.get(Session, index=0)[0]
        assert entity.path == 'subject_a/sess_0'


class TestContextManager:

    def test_exception_skips_commit(self, populated):
        entity = populated.get(Session)[0]

        with pytest.raises(ValueError):
            with entity:
                entity['partial'] = 1
                raise ValueError('boom')

        # The attribute must not have been persisted
        assert not populated.backend.has_entity_attribute(entity, 'partial')

    def test_normal_exit_commits(self, populated):
        entity = populated.get(Session)[0]
        with entity:
            entity['committed'] = 7

        entity._attribute_cache.clear()
        assert entity['committed'] == 7


class TestConsoleHelper:

    def test_ascii_fallback(self, monkeypatch):
        class FakeStdout:
            encoding = 'cp1252'
        monkeypatch.setattr(sys, 'stdout', FakeStdout())
        style = console.bar_style()
        assert style['spinner'] == 'classic'
        assert style['bar'] == 'classic'

    def test_unicode_keeps_fish(self, monkeypatch):
        class FakeStdout:
            encoding = 'utf-8'
        monkeypatch.setattr(sys, 'stdout', FakeStdout())
        style = console.bar_style()
        assert style['spinner'] == 'fish2'
        assert 'bar' not in style

    def test_overrides_win(self, monkeypatch):
        class FakeStdout:
            encoding = 'cp1252'
        monkeypatch.setattr(sys, 'stdout', FakeStdout())
        style = console.bar_style(bar=None, length=10)
        assert style['bar'] is None
        assert style['length'] == 10
