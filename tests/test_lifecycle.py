import builtins
import os
import random

import pytest
import yaml

from conftest import LabArchy, Session, Subject
from entarchy.backend import SQLiteBackend


class TestCreate:

    def test_create_returns_usable_instance(self, ent):
        # Regression: the instance returned by create() used to have a stale
        # in-memory config without entarchy_uuid, so .root raised RuntimeError
        assert ent.root is not None
        assert 'entarchy_uuid' in ent.get_config()

    def test_yaml_is_valid_and_complete(self, ent):
        config = yaml.safe_load(open(os.path.join(ent.path, 'entarchy.yaml')))
        assert config['entarchy_uuid'] == ent.root.uuid
        assert config['hierarchy'] == ent.hierarchy

    def test_reopen(self, ent):
        reopened = LabArchy(ent.path)
        assert reopened.root.uuid == ent.root.uuid
        reopened.backend.close()

    def test_create_refuses_existing_path(self, ent):
        with pytest.raises(FileExistsError):
            LabArchy.create(ent.path, SQLiteBackend(ent.path, dbname='other.db'))


class TestDelete:

    def test_wrong_verification_aborts(self, populated, monkeypatch):
        # Regression: a wrong verification string used to print "Abort." and
        # then delete everything anyway
        monkeypatch.setattr(builtins, 'input', lambda *a, **k: 'DEFINITELY-WRONG')

        populated.delete()

        assert os.path.exists(populated.path)
        assert os.path.exists(os.path.join(populated.path, 'entarchy.yaml'))
        assert os.path.exists(os.path.join(populated.path, 'test.db'))

        # Data is still intact
        reopened = LabArchy(populated.path)
        assert len(reopened.get(Session)) == 6
        reopened.backend.close()

    def test_correct_verification_deletes(self, populated, monkeypatch):
        monkeypatch.setattr(random, 'choices', lambda *a, **k: list('ABC12'))
        monkeypatch.setattr(builtins, 'input', lambda *a, **k: 'ABC12')

        populated.delete()

        assert not os.path.exists(populated.path)


class TestAnalysis:

    def test_analysis_uuid_stable_across_sessions(self, populated):
        populated.set_current_analysis('my_analysis')
        first_uuid = populated.current_analysis.uuid

        # Same instance, set again
        populated.set_current_analysis('my_analysis')
        assert populated.current_analysis.uuid == first_uuid

        # Fresh instance from disk
        reopened = LabArchy(populated.path)
        reopened.set_current_analysis('my_analysis')
        assert reopened.current_analysis.uuid == first_uuid
        reopened.backend.close()

    def test_attributes_reference_persisted_analysis(self, populated):
        populated.set_current_analysis('my_analysis')
        analysis_uuid = populated.current_analysis.uuid

        subject = populated.get(Subject)[0]
        subject['analyzed_value'] = 1.0

        import sqlite3
        con = sqlite3.connect(os.path.join(populated.path, 'test.db'))
        stored = con.execute(
            "SELECT analysis_uuid FROM attributes WHERE name = 'analyzed_value'").fetchone()[0]
        con.close()
        assert stored == analysis_uuid
