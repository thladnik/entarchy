"""Removing attributes and entities: docs/proposals/removing-things.md."""
import os

import numpy as np
import pytest
import sqlalchemy

from conftest import Animal, Layer, Recording, Roi, Session, Subject
from entarchy.backend.sql import AttributeTable, EntityTable, Link, _get_attribute_fp
from entarchy.core.entity import RemovalReport


def rows(ent, table, **filters) -> int:
    with sqlalchemy.orm.Session(ent.backend.sql_engine) as session:
        return session.query(table).filter_by(**filters).count()


def fresh(entity, name):
    entity._attribute_cache.pop(name, None)
    return entity[name]


class TestDeleteAttribute:

    def test_outside_a_context_the_row_goes_at_once(self, populated):
        subject = populated.get(Subject)[0]

        del subject['strain']

        assert 'strain' not in subject
        assert 'strain' not in subject.keys()
        with pytest.raises(AttributeError):            # as any missing attribute reads
            subject['strain']
        assert rows(populated, AttributeTable, entity_uuid=subject.uuid, name='strain') == 0

    def test_inside_a_context_it_waits_for_the_commit(self, populated):
        subject = populated.get(Subject)[0]

        with populated:
            del subject['strain']
            assert 'strain' not in subject
            assert 'strain' not in subject.keys()
            assert rows(populated, AttributeTable, entity_uuid=subject.uuid, name='strain') == 1

        assert rows(populated, AttributeTable, entity_uuid=subject.uuid, name='strain') == 0

    def test_a_write_after_a_queued_delete_cancels_it(self, populated):
        subject = populated.get(Subject)[0]

        with populated:
            del subject['strain']
            subject['strain'] = 'mutant'
            assert subject['strain'] == 'mutant'

        assert fresh(subject, 'strain') == 'mutant'

    def test_a_delete_after_a_queued_write_drops_the_write(self, populated):
        subject = populated.get(Subject)[0]

        with populated:
            subject['strain'] = 'mutant'
            del subject['strain']

        assert 'strain' not in subject
        assert rows(populated, AttributeTable, entity_uuid=subject.uuid, name='strain') == 0

    def test_a_failed_context_forgets_the_delete(self, populated):
        subject = populated.get(Subject)[0]

        with pytest.raises(ValueError):
            with populated:
                del subject['strain']
                raise ValueError('boom')

        assert subject['strain'] == 'wildtype'
        assert subject._attributes_to_delete == []

    def test_missing_is_a_key_error_and_pop_has_a_default(self, populated):
        subject = populated.get(Subject)[0]

        with pytest.raises(KeyError):
            del subject['nope']
        assert subject.pop('nope', None) is None
        with pytest.raises(KeyError):
            subject.pop('nope')

        assert subject.pop('strain') == 'wildtype'
        assert 'strain' not in subject

    def test_identity_is_never_removable(self, populated):
        subject = populated.get(Subject)[0]

        for name in ('id', 'uuid'):
            with pytest.raises(RuntimeError, match='identity'):
                del subject[name]

    def test_immutable_attributes_follow_the_overwrite_rule(self, ent):
        subject = ent.add_child(Subject, 'a')
        ent.start_digest()
        subject['strain'] = 'wt'
        ent.end_digest()

        with pytest.raises(RuntimeError, match='immutable'):
            del subject['strain']
        assert fresh(subject, 'strain') == 'wt'

        ent.start_digest()
        del subject['strain']
        ent.end_digest()
        assert 'strain' not in subject

    def test_a_value_kept_in_a_file_takes_the_file_with_it(self, ent):
        subject = ent.add_child(Subject, 'a')
        ent.max_blob_size = 1024
        subject['trace'] = np.arange(10_000.0)
        directory, filename = _get_attribute_fp(ent.path, subject.uuid, 'trace', 'blob')
        path = os.path.join(directory, filename)
        assert os.path.exists(path)

        del subject['trace']

        assert not os.path.exists(path)

    def test_the_file_goes_only_once_the_context_commits(self, ent):
        subject = ent.add_child(Subject, 'a')
        ent.max_blob_size = 1024
        subject['trace'] = np.arange(10_000.0)
        path = os.path.join(*_get_attribute_fp(ent.path, subject.uuid, 'trace', 'blob'))

        with ent:
            del subject['trace']
            assert os.path.exists(path)

        assert not os.path.exists(path)

    def test_an_unbound_entity_forgets_the_cached_write(self):
        subject = Subject('a')
        subject['strain'] = 'wt'

        del subject['strain']

        assert 'strain' not in subject
        assert subject.keys() == []
        with pytest.raises(KeyError):
            del subject['strain']

    def test_a_pending_entity_forgets_the_write(self, ent):
        with ent:
            subject = ent.add_child(Subject, 'a')
            subject['strain'] = 'wt'
            del subject['strain']

        assert 'strain' not in subject
        assert rows(ent, AttributeTable, entity_uuid=subject.uuid, name='strain') == 0


class TestDeleteCollectionAttribute:

    def test_removes_the_attribute_from_every_entity(self, populated):
        sessions = populated.get(Session)
        assert len(sessions['flag']) == 6

        del sessions['flag']

        assert rows(populated, AttributeTable, name='flag') == 0
        assert all('flag' not in session for session in sessions)
        assert 'flag' not in sessions._cache.columns

    def test_only_from_the_entities_the_collection_matched(self, populated):
        del populated.get(Session, 'index > 2')['flag']

        assert rows(populated, AttributeTable, name='flag') == 3
        assert sorted(s['index'] for s in populated.get(Session, 'flag == True')) == [0, 2]

    def test_a_name_nobody_has_is_a_key_error(self, populated):
        with pytest.raises(KeyError):
            del populated.get(Session)['nope']

    def test_identity_is_refused(self, populated):
        with pytest.raises(RuntimeError, match='identity'):
            del populated.get(Session)['id']

    def test_immutable_rows_refuse_the_whole_collection(self, ent):
        subject = ent.add_child(Subject, 'a')
        plain = subject.add_child(Session, 'plain')
        plain['score'] = 1.0
        ent.start_digest()
        ingested = subject.add_child(Session, 'ingested')
        ingested['score'] = 2.0
        ent.end_digest()

        with pytest.raises(RuntimeError, match='immutable on 1 of the 2'):
            del ent.get(Session)['score']
        assert rows(ent, AttributeTable, name='score') == 2

        ent.start_digest()
        del ent.get(Session)['score']
        ent.end_digest()
        assert rows(ent, AttributeTable, name='score') == 0


class TestRemoveChild:

    def test_a_childless_entity(self, deep):
        roi = deep.get(Roi)[0]
        layer = roi.parent

        report = layer.remove_child(roi)

        assert isinstance(report, RemovalReport)
        assert report.entities == {'Roi': 1}
        assert report.attributes == 3                      # id, uuid, index
        assert len(deep.get(Roi)) == 11
        assert rows(deep, EntityTable, uuid=roi.uuid) == 0
        assert rows(deep, AttributeTable, entity_uuid=roi.uuid) == 0

    def test_children_block_unless_asked(self, deep):
        recording = deep.get(Recording)[0]
        animal = recording.parent

        with pytest.raises(ValueError, match='2 Layer, 6 Roi below it'):
            animal.remove_child(recording)
        assert len(deep.get(Roi)) == 12

        report = animal.remove_child(recording, recursive=True)

        assert report.entities == {'Recording': 1, 'Layer': 2, 'Roi': 6}
        assert len(deep.get(Recording)) == 1
        assert len(deep.get(Layer)) == 2
        assert len(deep.get(Roi)) == 6
        assert rows(deep, EntityTable) == 1 + 1 + 1 + 2 + 6   # root, animal, recording, layers, rois

    def test_dry_run_changes_nothing(self, deep):
        recording = deep.get(Recording)[0]

        report = recording.parent.remove_child(recording, recursive=True, dry_run=True)

        assert report.entities == {'Recording': 1, 'Layer': 2, 'Roi': 6}
        assert len(deep.get(Roi)) == 12
        assert not recording.is_removed
        assert 'Roi' in str(report)

    def test_links_go_with_either_end_and_do_not_count_as_children(self, deep):
        deep.define_link_type('mean_response', Recording, Roi)
        rec_0, rec_1 = sorted(deep.get(Recording), key=lambda r: r.id)
        roi_of_0 = rec_0.get(Roi)[0]
        roi_of_1 = rec_1.get(Roi)[0]
        deep.link(rec_0, roi_of_1, 'mean_response', mean_dff=0.1)   # rec_0 is the linker
        deep.link(rec_1, roi_of_0, 'mean_response', mean_dff=0.2)   # rec_0's roi is linked
        assert rows(deep, Link) == 2

        report = rec_0.parent.remove_child(rec_0, recursive=True)

        assert report.links == {'mean_response': 2}
        assert rows(deep, Link) == 0
        assert len(deep.get('LinkEntity')) == 0
        assert len(rec_1.get(Roi)) == 6

    def test_a_link_carrier_is_not_a_child(self, deep):
        deep.define_link_type('correlated', Roi, Roi, symmetric=True)
        roi_a, roi_b = deep.get(Roi)[0], deep.get(Roi)[1]
        deep.link(roi_a, roi_b, 'correlated')

        report = roi_a.parent.remove_child(roi_a)            # not recursive

        assert report.links == {'correlated': 1}
        assert rows(deep, Link) == 0
        assert len(deep.get(Roi)) == 11

    def test_the_wrong_parent_is_refused(self, deep):
        roi = deep.get(Roi)[0]
        other_layer = [layer for layer in deep.get(Layer) if layer.uuid != roi.parent.uuid][0]

        with pytest.raises(ValueError, match='is a child of'):
            other_layer.remove_child(roi)
        assert len(deep.get(Roi)) == 12

    def test_the_object_is_dead_afterwards(self, deep):
        roi = deep.get(Roi)[0]
        roi.parent.remove_child(roi)

        assert roi.is_removed
        assert not roi.is_bound
        assert roi not in deep
        with pytest.raises(RuntimeError, match='removed'):
            roi['index']
        with pytest.raises(RuntimeError, match='removed'):
            roi['index'] = 1
        with pytest.raises(RuntimeError, match='removed'):
            'index' in roi

    def test_the_id_is_free_again(self, deep):
        roi = deep.get(Roi)[0]
        layer = roi.parent
        layer.remove_child(roi)

        again = layer.add_child(Roi, roi.id, index=99)

        assert again.uuid != roi.uuid
        assert len(deep.get(Roi)) == 12

    def test_an_ingested_entity_needs_digest_mode(self, ent):
        ent.start_digest()
        subject = ent.add_child(Subject, 'a', strain='wt')
        ent.end_digest()

        with pytest.raises(RuntimeError, match='digest mode'):
            ent.remove_child(subject)
        assert len(ent.get(Subject)) == 1
        assert ent.remove_child(subject, dry_run=True).immutable == 3

        ent.start_digest()
        ent.remove_child(subject)
        ent.end_digest()
        assert len(ent.get(Subject)) == 0

    def test_a_pending_subtree_is_dequeued(self, ent):
        with ent:
            subject = ent.add_child(Subject, 'a')
            session = subject.add_child(Session, 's', index=1)

            with pytest.raises(ValueError, match='1 Session'):
                ent.remove_child(subject)

            report = ent.remove_child(subject, recursive=True)
            assert report.entities == {'Subject': 1, 'Session': 1}
            assert subject.is_removed and session.is_removed

        assert len(ent.get(Subject)) == 0
        assert ent._entities_to_add == [] and ent._entities_to_update == []

    def test_a_pending_child_of_a_stored_entity_goes_too(self, populated):
        subject = populated.get(Subject)[0]

        with populated:
            subject.add_child(Session, 'queued')
            populated.remove_child(subject, recursive=True)

        assert len(populated.get(Subject)) == 0
        assert len(populated.get(Session)) == 0

    def test_queued_writes_on_removed_entities_are_dropped(self, populated):
        subject = populated.get(Subject)[0]
        session = subject.get(Session)[0]

        with populated:
            session['index'] = 99
            populated.remove_child(subject, recursive=True)

        assert populated._entities_to_update == []
        assert rows(populated, EntityTable) == 1

    def test_files_are_removed(self, deep):
        roi = deep.get(Roi)[0]
        deep.max_blob_size = 1024
        roi['trace'] = np.arange(10_000.0)
        path = os.path.join(*_get_attribute_fp(deep.path, roi.uuid, 'trace', 'blob'))
        assert os.path.exists(path)

        report = roi.parent.remove_child(roi)

        assert [os.path.normpath(f) for f in report.files] == [os.path.normpath(path)]
        assert not os.path.exists(path)

    def test_the_entarchy_forwards_to_the_root(self, populated):
        subject = populated.get(Subject)[0]

        report = populated.remove_child(subject, recursive=True)

        assert report.entities == {'Subject': 1, 'Session': 6}
        assert len(populated.get(Subject)) == 0
        assert len(populated.get(Session)) == 0

    def test_built_in_entities_stay(self, ent):
        with pytest.raises(ValueError, match='built-in'):
            ent.root.remove_child(ent.root)

    def test_the_wrong_type_is_refused(self, populated):
        with pytest.raises(TypeError, match='Entity'):
            populated.remove_child(Subject.where('id == "x"'))
