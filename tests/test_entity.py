import sys

import pytest

from conftest import Session, Subject
from entarchy.core import console


class TestIdentityMap:

    def test_same_uuid_returns_same_object(self, populated):
        first = populated.get(Session)[0]
        again = Session(populated, _uuid=first.uuid, _id=first.id)
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
            same = Session(populated, _uuid=entity.uuid, _id=entity.id,
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
