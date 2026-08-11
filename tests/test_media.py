"""Media attributes: files entarchy keeps but never reads.

The property that drives all of it is that an entarchy is self-contained, so
most of these tests are about the file being inside the directory and staying
reachable when the directory moves, is archived, or is imported back.
"""
import os
import shutil

import numpy as np
import pytest

import entarchy
from entarchy import MediaFile
from entarchy.backend import SQLiteBackend
from entarchy.backend.sql import _get_media_fp

from conftest import LabArchy, Session, Subject


@pytest.fixture()
def source_file(tmp_path):
    path = tmp_path / 'sources' / 'behaviour.avi'
    path.parent.mkdir()
    path.write_bytes(b'FAKE-AVI' + bytes(range(256)) * 40)
    return path


@pytest.fixture()
def subject(populated):
    return populated.get(Subject)[0]


class TestStoring:

    def test_assignment_copies_the_file_in(self, populated, subject, source_file):
        subject['video'] = MediaFile(source_file)

        stored = subject['video']
        assert stored.is_stored
        assert stored.relative_path.startswith('media/')
        assert stored.path.startswith(populated.path)
        assert stored.read_bytes() == source_file.read_bytes()

    def test_set_media_does_the_same(self, subject, source_file):
        stored = subject.set_media('video', source_file)

        assert stored.is_stored
        assert stored.read_bytes() == source_file.read_bytes()

    def test_the_source_is_left_alone(self, subject, source_file):
        subject['video'] = MediaFile(source_file)

        assert source_file.exists()

    def test_move_consumes_the_source(self, subject, source_file):
        subject['video'] = MediaFile(source_file, move=True)

        assert not source_file.exists()
        assert subject['video'].exists()

    def test_reading_back_gives_the_stored_file_not_the_source(self, subject, source_file):
        """The cache must not keep the source handle: it names a file outside the
        entarchy that entarchy does not own and may not keep."""
        subject['video'] = MediaFile(source_file)

        assert subject['video'].is_stored
        assert str(source_file) not in subject['video'].path

    def test_a_missing_source_is_refused(self, subject, tmp_path):
        with pytest.raises(FileNotFoundError, match='does not exist'):
            subject['video'] = MediaFile(tmp_path / 'nothing.avi')

    def test_media_type_is_guessed_and_can_be_given(self, subject, source_file):
        subject['guessed'] = MediaFile(source_file)
        subject['explicit'] = MediaFile(source_file, media_type='video/x-custom')

        assert subject['guessed'].media_type.startswith('video/')
        assert subject['explicit'].media_type == 'video/x-custom'

    def test_an_unknown_extension_still_stores(self, subject, tmp_path):
        odd = tmp_path / 'thing.zzz'
        odd.write_bytes(b'payload')

        subject['odd'] = MediaFile(odd)

        assert subject['odd'].media_type == 'application/octet-stream'
        assert subject['odd'].read_bytes() == b'payload'


class TestReading:

    def test_is_os_pathlike(self, subject, source_file):
        subject['video'] = MediaFile(source_file)

        assert os.fspath(subject['video']) == subject['video'].path
        assert open(subject['video'], 'rb').read(8) == b'FAKE-AVI'

    def test_open_and_read_bytes(self, subject, source_file):
        subject['video'] = MediaFile(source_file)

        with subject['video'].open() as f:
            assert f.read(8) == b'FAKE-AVI'

    def test_digest_verifies(self, subject, source_file):
        subject['video'] = MediaFile(source_file)

        assert subject['video'].sha256 is not None
        assert subject['video'].verify()

    def test_a_changed_file_fails_verification(self, subject, source_file):
        subject['video'] = MediaFile(source_file)

        with open(subject['video'].path, 'ab') as f:
            f.write(b'tampered')

        assert not subject['video'].verify()

    def test_a_missing_file_does_not_break_the_read(self, subject, source_file):
        """Reading builds the handle from the row and touches no file, so a
        DataFrame over a thousand entities does not fail on one absent video."""
        subject['video'] = MediaFile(source_file)
        os.remove(subject['video'].path)

        stored = subject['video']

        assert not stored.exists()
        assert not stored.verify()
        with pytest.raises(FileNotFoundError):
            stored.open()

    def test_discovery_lists_media_attributes(self, subject, source_file):
        subject['video'] = MediaFile(source_file)
        subject['trace'] = np.arange(100)
        subject['count'] = 5

        assert subject.media() == ['video']

    def test_discovery_does_not_read_other_blobs(self, subject, source_file, tmp_path):
        """It inspects pointers only - asking self[name] for each would decode
        every payload the entity holds."""
        subject['video'] = MediaFile(source_file)
        subject['big'] = np.zeros(50_000)
        subject._attribute_cache.clear()

        assert subject.media() == ['video']
        assert 'big' not in subject._attribute_cache


class TestLayout:

    def test_the_file_lives_under_the_entarchy(self, populated, subject, source_file):
        subject['video'] = MediaFile(source_file)

        relative = subject['video'].relative_path
        assert not os.path.isabs(relative)
        assert os.path.exists(os.path.join(populated.path, relative))

    def test_the_name_comes_from_the_attribute_not_the_source(self, subject, source_file):
        """An acquisition file name can be seventy characters of instrument
        settings; keeping it would make every media path as long as whatever
        the microscope wrote."""
        from entarchy.backend.sql import _get_namehash

        subject['video'] = MediaFile(source_file)

        assert subject['video'].relative_path.endswith(f'{_get_namehash("video")}.avi')
        assert 'behaviour' not in subject['video'].relative_path

    def test_two_attributes_do_not_collide(self, subject, source_file):
        subject['video_a'] = MediaFile(source_file)
        subject['video_b'] = MediaFile(source_file)

        assert subject['video_a'].relative_path != subject['video_b'].relative_path
        assert subject['video_a'].exists() and subject['video_b'].exists()

    def test_the_path_length_does_not_depend_on_the_source(self, tmp_path):
        """Windows counts every character against a 260 limit, so the path a
        media file gets has to be bounded whatever it was called."""
        short = _get_media_fp(str(tmp_path), 'aaaabbbb-cccc-dddd-eeee-ffff00001111',
                              'some/attribute', 'a.tif')
        long_name = ('2026-01-15_fish1_5dpf_jf7_14laser_651gain_2hz_2p6mag_'
                     '00001_cropped_0001.tif')
        long = _get_media_fp(str(tmp_path), 'aaaabbbb-cccc-dddd-eeee-ffff00001111',
                             'some/attribute', long_name)

        assert short == long

        relative = os.path.relpath(os.path.join(*long), str(tmp_path))
        assert len(relative) < 80

    def test_a_hostile_source_name_cannot_reach_the_path(self, subject, tmp_path):
        odd = tmp_path / 'we ird (name)!.avi'
        odd.write_bytes(b'x')

        subject['video'] = MediaFile(odd)

        stored = subject['video']
        assert stored.relative_path.endswith('.avi')
        for character in ' ()!':
            assert character not in stored.relative_path
        assert stored.read_bytes() == b'x' 


class TestReplacement:

    def test_overwriting_with_the_same_kind_replaces_in_place(self, subject,
                                                              source_file, tmp_path):
        """The name comes from the attribute, so a replacement of the same kind
        lands on the same path - there is nothing left over to clean up."""
        subject['video'] = MediaFile(source_file)
        first = subject['video'].path

        replacement = tmp_path / 'sources' / 'behaviour_v2.avi'
        replacement.write_bytes(b'SECOND')
        subject['video'] = MediaFile(replacement)

        assert subject['video'].path == first
        assert subject['video'].read_bytes() == b'SECOND'
        assert subject['video'].verify()

    def test_overwriting_with_another_format_removes_the_old_file(self, subject,
                                                                  source_file, tmp_path):
        """A different extension is a different path, so the old file would be
        left behind."""
        subject['video'] = MediaFile(source_file)
        first = subject['video'].path

        replacement = tmp_path / 'sources' / 'behaviour.mp4'
        replacement.write_bytes(b'SECOND')
        subject['video'] = MediaFile(replacement)

        assert subject['video'].path != first
        assert not os.path.exists(first)
        assert subject['video'].read_bytes() == b'SECOND'

    def test_replacing_media_with_an_array_removes_the_file(self, subject, source_file):
        subject['video'] = MediaFile(source_file)
        path = subject['video'].path

        subject['video'] = np.arange(10)

        assert not os.path.exists(path)
        assert list(subject['video']) == list(range(10))

    def test_a_shrinking_blob_does_not_leave_its_file(self, populated, subject):
        """The ext/ payload of a large value is orphaned when the next value is
        small enough to sit in the row.

        Random rather than zeros: an array of zeros compresses to well under the
        threshold and never reaches a file in the first place.
        """
        populated.max_blob_size = 1024
        subject['payload'] = np.random.default_rng(0).random(5000)

        store = entarchy.backend.blob_store.store_of(_raw(populated, subject, 'payload'))
        assert store.startswith('ext/'), f'expected an external payload, got {store}'
        path = os.path.join(populated.path, store)
        assert os.path.exists(path)

        subject['payload'] = np.zeros(2)

        assert not os.path.exists(path)
        assert list(subject['payload']) == [0.0, 0.0]


def _raw(ent, entity, name):
    import sqlalchemy
    from sqlalchemy.orm import Session as SASession

    from entarchy.backend.sql import AttributeTable

    with SASession(ent.backend.sql_engine) as session:
        return session.query(AttributeTable.value_blob).filter(
            AttributeTable.entity_uuid == entity.uuid,
            AttributeTable.name == name).scalar()


class TestAccounting:

    def test_data_size_is_the_file_not_the_pointer(self, populated, subject, source_file):
        subject['video'] = MediaFile(source_file)

        raw = _raw(populated, subject, 'video')
        import sqlalchemy
        from sqlalchemy.orm import Session as SASession

        from entarchy.backend.sql import AttributeTable

        with SASession(populated.backend.sql_engine) as session:
            size = session.query(AttributeTable.data_size).filter(
                AttributeTable.entity_uuid == subject.uuid,
                AttributeTable.name == 'video').scalar()

        assert size == source_file.stat().st_size
        assert len(raw) < 400


class TestSelfContained:

    def test_the_entarchy_can_be_moved(self, populated, subject, source_file, tmp_path):
        subject['video'] = MediaFile(source_file)
        original = populated.path
        populated.backend.close()

        moved = (tmp_path / 'moved').as_posix()
        shutil.move(original, moved)

        reopened = LabArchy(moved)
        try:
            stored = reopened.get(Subject)[0]['video']
            assert stored.exists()
            assert stored.verify()
            assert stored.path.startswith(moved)
        finally:
            reopened.backend.close()

    def test_nothing_outside_the_entarchy_is_referenced(self, populated, subject, source_file):
        subject['video'] = MediaFile(source_file)

        relative = subject['video'].relative_path
        assert not os.path.isabs(relative)
        assert '..' not in relative


class TestCollectionWrites:

    def test_media_through_a_collection(self, populated, tmp_path):
        import pandas as pd

        sessions = populated.get(Session)
        sources = {}
        for index, uuid in enumerate(sessions.index):
            path = tmp_path / f'clip_{index}.avi'
            path.write_bytes(f'CLIP{index}'.encode())
            sources[uuid] = MediaFile(path)

        sessions['clip'] = pd.Series(sources)

        for index, entity in enumerate(sorted(populated.get(Session), key=lambda e: e.id)):
            assert entity['clip'].exists()
            assert entity['clip'].read_bytes().startswith(b'CLIP')
