"""Digest mode: raw-data immutability and cache purging during bulk ingest."""
import numpy as np
import pytest

import entarchy
from conftest import Session, Subject


@entarchy.digest_method
def ingest(ent, value=1):
    with ent:
        subject = Subject(ent, _id='ingested', _parent=ent.root)
        ent.add_new_entity(subject)
        subject['raw'] = value
    return subject


@entarchy.digest_method
def failing_ingest(ent):
    raise ValueError('ingest blew up')


@entarchy.digest_method
def rewrite(ent, subject, value):
    subject['raw'] = value


class TestDigestModeFlag:

    def test_flag_toggles(self, ent):
        assert not ent.is_in_digest_mode
        ent.start_digest()
        assert ent.is_in_digest_mode
        ent.end_digest()
        assert not ent.is_in_digest_mode

    def test_decorator_sets_and_restores(self, ent):
        assert not ent.is_in_digest_mode
        ingest(ent)
        assert not ent.is_in_digest_mode

    def test_decorator_restores_on_exception(self, ent):
        with pytest.raises(ValueError, match='blew up'):
            failing_ingest(ent)
        assert not ent.is_in_digest_mode


class TestImmutability:

    def test_ingested_attributes_are_immutable_afterwards(self, ent):
        subject = ingest(ent, value=1)

        with pytest.raises(RuntimeError, match='immutable'):
            subject['raw'] = 2

        subject._attribute_cache.clear()
        assert subject['raw'] == 1

    def test_digest_mode_may_rewrite_ingested_attributes(self, ent):
        subject = ingest(ent, value=1)

        rewrite(ent, subject, 99)

        subject._attribute_cache.clear()
        assert subject['raw'] == 99

    def test_attributes_written_outside_digest_stay_mutable(self, populated):
        entity = populated.get(Session)[0]
        entity['analysis_result'] = 1
        entity['analysis_result'] = 2  # must not raise

        entity._attribute_cache.clear()
        assert entity['analysis_result'] == 2


class TestCachePurging:

    def test_cache_is_purged_on_commit_in_digest_mode(self, ent):
        with ent:
            subject = Subject(ent, _id='s', _parent=ent.root)
            ent.add_new_entity(subject)
            subject['big'] = np.zeros(100)

        # Outside digest mode the cache is retained
        assert 'big' in subject._attribute_cache

        ent.start_digest()
        try:
            subject.commit()
        finally:
            ent.end_digest()

        assert subject._attribute_cache == {}
        # ... and the value is still readable from the backend
        assert np.array_equal(subject['big'], np.zeros(100))

    def test_digest_ingest_leaves_no_cached_attributes(self, ent):
        subject = ingest(ent)
        assert subject._attribute_cache == {}
