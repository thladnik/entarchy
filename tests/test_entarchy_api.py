"""Entarchy object API: config validation, registry, pickling and commit batching."""
import os
import pickle

import numpy as np
import pytest
import yaml

from conftest import LabArchy, Session, Subject


def rewrite_config(path, **changes):
    config_path = os.path.join(path, 'entarchy.yaml')
    config = yaml.safe_load(open(config_path))
    config.update(changes)
    with open(config_path, 'w') as f:
        yaml.safe_dump(config, f)


class TestConfigValidation:

    def test_incompatible_base_version_raises(self, ent):
        """The storage format is not backward compatible, so this is what makes
        an unreadable directory say so rather than fail deep inside a value read."""
        rewrite_config(ent.path, base_version='0.0')
        with pytest.raises(RuntimeError, match='storage format changed'):
            LabArchy(ent.path)

    def test_the_version_message_names_the_directory_and_both_versions(self, ent):
        rewrite_config(ent.path, base_version='0.1')

        with pytest.raises(RuntimeError) as raised:
            LabArchy(ent.path)

        assert '0.1' in str(raised.value)
        assert LabArchy._base_version in str(raised.value)
        assert ent.path in str(raised.value)

    def test_a_new_entarchy_records_the_current_base_version(self, ent):
        config = yaml.safe_load(open(os.path.join(ent.path, 'entarchy.yaml')))

        assert config['base_version'] == LabArchy._base_version

    def test_incompatible_implementation_version_raises(self, ent):
        rewrite_config(ent.path, implementation_version='99.9')
        with pytest.raises(RuntimeError, match='Implementation version'):
            LabArchy(ent.path)

    def test_hierarchy_mismatch_raises(self, ent):
        rewrite_config(ent.path, hierarchy={'Subject': {}})
        with pytest.raises(RuntimeError, match='hierarchy'):
            LabArchy(ent.path)

    def test_missing_config_raises(self, tmp_path):
        empty = tmp_path / 'empty'
        empty.mkdir()
        with pytest.raises(FileNotFoundError):
            LabArchy(empty.as_posix())

    def test_get_config_returns_a_copy(self, ent):
        config = ent.get_config()
        config['base_version'] = 'tampered'
        assert ent.get_config()['base_version'] != 'tampered'


class TestEntityTypes:

    def test_lookup_by_name(self, ent):
        assert ent.get_entity_type('Subject') is Subject
        assert ent.get_entity_type('Session') is Session

    def test_unknown_type_raises(self, ent):
        with pytest.raises(ValueError, match='not found in hierarchy'):
            ent.get_entity_type('Ghost')

    def test_get_accepts_type_name_string(self, populated):
        assert len(populated.get('Session')) == len(populated.get(Session))

    def test_hierarchy_includes_builtin_types(self, ent):
        hierarchy = ent.hierarchy
        assert 'Subject' in hierarchy
        assert hierarchy['Subject'] == {'Session': {}}
        for builtin in ('EntarchyEntity', 'AnalysisEntity', 'LinkEntity'):
            assert builtin in hierarchy

    def test_hierarchy_property_returns_a_copy(self, ent):
        ent.hierarchy['Subject'] = 'tampered'
        assert ent.hierarchy['Subject'] != 'tampered'

    def test_duplicate_entity_names_rejected(self):
        import entarchy

        class Dup(entarchy.Entity):
            pass

        Dup.add_child_entity_type(Dup)  # self-reference -> duplicate name

        class BadArchy(entarchy.Entarchy):
            _implementation_version = '0.1'
            _implementation_compat_version_list = ['0.1']
            _hierarchy_root_type = Dup

        with pytest.raises(ValueError, match='already in the hierarchy'):
            BadArchy._resolve_hierarchy()


class TestRegistry:

    def test_contains_entity_and_uuid(self, populated):
        entity = populated.get(Session)[0]
        assert entity in populated
        assert entity.uuid in populated
        assert 'not-a-uuid' not in populated
        assert 12345 not in populated

    def test_get_entity_by_uuid_uses_registry(self, populated):
        entity = populated.get(Session)[0]
        assert populated.get_entity_by_uuid(entity.uuid) is entity

    def test_get_entity_by_uuid_loads_unknown(self, populated):
        entity = populated.get(Session)[0]
        uuid = entity.uuid
        populated._entities.clear()

        loaded = populated.get_entity_by_uuid(uuid)
        assert loaded.uuid == uuid
        assert isinstance(loaded, Session)

    def test_unknown_uuid_raises(self, populated):
        with pytest.raises(RuntimeError, match='does not exist'):
            populated.get_entity_by_uuid('aaaaaaaa-0000-0000-0000-000000000000')


class TestClearRegistry:

    def test_releases_cached_entities(self, populated):
        entities = list(populated.get(Session))
        assert len(populated._entities) >= len(entities)

        released = populated.clear_registry()

        assert released >= len(entities)
        assert len(populated._entities) == 0

    def test_released_entities_are_reloaded_on_demand(self, populated):
        entity = populated.get(Session)[0]
        uuid, index = entity.uuid, entity['index']

        populated.clear_registry()

        reloaded = populated.get_entity_by_uuid(uuid)
        assert reloaded.uuid == uuid
        assert reloaded['index'] == index

    def test_frees_cached_attribute_data(self, populated):
        entity = populated.get(Session)[0]
        entity['big'] = np.zeros(1000)
        assert 'big' in entity._attribute_cache

        populated.clear_registry()

        # The registry no longer holds the entity, so its cache can be collected
        assert entity.uuid not in populated._entities

    def test_keeps_entities_with_uncommitted_changes(self, populated):
        with populated:
            entity = populated.get(Session)[0]
            entity['pending'] = 1

            populated.clear_registry()

            assert entity.uuid in populated._entities

        # The pending write survived and was committed on context exit
        entity._attribute_cache.clear()
        assert entity['pending'] == 1

    def test_keeps_entities_queued_for_insertion(self, ent):
        with ent:
            subject = Subject(ent, _id='new', _parent=ent.root)
            ent.add_new_entity(subject)

            ent.clear_registry()
            assert subject.uuid in ent._entities

        subject._attribute_cache.clear()
        assert subject['id'] == 'new'

    def test_is_safe_on_an_empty_registry(self, ent):
        ent.clear_registry()
        assert ent.clear_registry() >= 0


class TestPickling:

    def test_getstate_omits_backend_and_registry(self, populated):
        # Ensure backend and registry are populated before inspecting the state
        assert len(populated.get(Session)) == 6
        state = populated.__getstate__()

        assert '_backend' not in state
        assert state['_entities'] == {}
        assert state['_entities_to_add'] == []
        assert state['_entities_to_update'] == []

    def test_state_size_is_independent_of_entity_count(self, populated):
        small = len(pickle.dumps(populated))
        for entity in populated.get(Session):
            _ = entity['index']  # force them all into the registry
        assert len(populated._entities) >= 6
        assert len(pickle.dumps(populated)) == small

    def test_roundtrip_stays_usable(self, populated):
        restored = pickle.loads(pickle.dumps(populated))
        assert restored.path == populated.path
        assert len(restored.get(Session)) == 6
        restored.backend.close()

    def test_original_backend_survives_pickling(self, populated):
        pickle.dumps(populated)
        assert len(populated.get(Session)) == 6


class TestCommitBatching:

    def test_context_defers_writes(self, ent):
        with ent:
            subject = Subject(ent, _id='batched', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['a'] = 1
            assert subject.uuid in ent._entities_to_update

        assert ent._entities_to_update == []
        assert ent._entities_to_add == []

    def test_write_outside_context_commits_immediately(self, populated):
        entity = populated.get(Session)[0]
        entity['immediate'] = 5
        assert entity._attributes_to_update == []
        assert populated.backend.has_entity_attribute(entity, 'immediate')

    def test_remove_entity_from_update(self, ent):
        with ent:
            subject = Subject(ent, _id='x', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['a'] = 1
            ent.remove_entity_from_update(subject)
            assert subject.uuid not in ent._entities_to_update

    def test_entity_context_nested_in_entarchy_context(self, ent):
        """Regression: the inner entity context used to flush attributes before the
        entity row existed, raising NoResultFound."""
        with ent:
            subject = Subject(ent, _id='nested', _parent=ent.root)
            ent.add_new_entity(subject)
            with subject:
                subject['a'] = 1
            # Still deferred - the entity row is only inserted on the outer exit
            assert subject.uuid in ent._entities_to_add
            assert 'a' in subject._attributes_to_update

        assert subject._attributes_to_update == []
        subject._attribute_cache.clear()
        assert subject['a'] == 1

    def test_entity_context_outside_entarchy_context(self, populated):
        entity = populated.get(Session)[0]
        with entity:
            entity['b'] = 2
            assert 'b' in entity._attributes_to_update
        assert entity._attributes_to_update == []


class TestTempPath:

    def test_creates_directory_under_entarchy(self, ent):
        temp = ent.get_temp_path('worker_1')
        assert os.path.isdir(temp)
        assert os.path.commonpath([temp, ent.path]) == os.path.normpath(ent.path)

    def test_is_idempotent(self, ent):
        assert ent.get_temp_path('again') == ent.get_temp_path('again')


class TestCollectionMisc:

    def test_columns_and_keys(self, populated):
        sessions = populated.get(Session)
        assert set(sessions.keys()) == set(sessions.columns)
        for expected in ('id', 'uuid', 'index', 'score', 'flag'):
            assert expected in sessions.columns

    def test_to_dict_generator(self, populated):
        records = list(populated.get(Session).to_dict())
        assert len(records) == 6
        assert all(r['id'].startswith('sess_') for r in records)

    def test_repr_shows_count(self, populated):
        assert 'count=6' in repr(populated.get(Session))

    def test_empty_collection(self, populated):
        empty = populated.get(Session, 'index > 100')
        assert len(empty) == 0
        assert list(empty) == []
        assert empty.map(lambda e: e) == []
        empty.map_async(len)  # must return early, not spawn a pool

    def test_invalid_key_raises(self, populated):
        with pytest.raises(KeyError):
            populated.get(Session)[1.5]
