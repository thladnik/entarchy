"""Migrating an entarchy created before link types existed."""
import os

import pytest
import sqlalchemy

from entarchy.tools import migrate_links

from conftest import DeepArchy, Recording, Roi


def _downgrade_to_old_schema(ent):
    """Recreate the link tables as a pre-link-type entarchy had them."""
    engine = ent.backend.sql_engine
    with engine.begin() as connection:
        connection.execute(sqlalchemy.text('DROP TABLE IF EXISTS links'))
        connection.execute(sqlalchemy.text('DROP TABLE IF EXISTS link_types'))
        connection.execute(sqlalchemy.text(
            'CREATE TABLE links ('
            ' linker_uuid VARCHAR(36) NOT NULL,'
            ' linked_uuid VARCHAR(36) NOT NULL,'
            ' entity_uuid VARCHAR(36),'
            ' created DATETIME,'
            ' modified DATETIME,'
            ' PRIMARY KEY (linker_uuid, linked_uuid))'))


def _url(ent):
    return f'sqlite:///{ent.path}/{ent.backend.dbname}'


@pytest.fixture()
def outdated(deep):
    _downgrade_to_old_schema(deep)
    deep.backend.close()
    return deep


class TestInspect:

    def test_detects_the_old_schema(self, outdated):
        state = migrate_links.inspect(_url(outdated))

        assert state['has_links']
        assert not state['has_link_types']
        assert not state['links_current']
        assert state['needs_migration']

    def test_a_current_entarchy_needs_nothing(self, deep):
        state = migrate_links.inspect(_url(deep))

        assert state['links_current']
        assert state['has_link_types']
        assert not state['needs_migration']


class TestMigrate:

    def test_dry_run_changes_nothing(self, outdated):
        assert migrate_links.migrate(_url(outdated), apply_changes=False, verbose=False)

        state = migrate_links.inspect(_url(outdated))
        assert state['needs_migration']

    def test_apply_creates_the_current_tables(self, outdated):
        migrate_links.migrate(_url(outdated), apply_changes=True, verbose=False)

        state = migrate_links.inspect(_url(outdated))
        assert state['links_current']
        assert state['has_link_types']
        assert not state['needs_migration']

    def test_migrated_entarchy_takes_link_types(self, outdated):
        migrate_links.migrate(_url(outdated), apply_changes=True, verbose=False)

        reopened = DeepArchy(outdated.path)
        try:
            reopened.define_link_type('mean_response', Recording, Roi)
            assert reopened.get_link_type('mean_response').linked.entity_type == 'Roi'
        finally:
            reopened.backend.close()

    def test_running_twice_is_harmless(self, outdated):
        migrate_links.migrate(_url(outdated), apply_changes=True, verbose=False)

        assert not migrate_links.migrate(_url(outdated), apply_changes=True, verbose=False)

    def test_refuses_to_drop_a_table_holding_rows(self, outdated):
        """Nothing ever wrote to the old table, so rows mean something unexpected."""
        engine = sqlalchemy.create_engine(_url(outdated))
        with engine.begin() as connection:
            connection.execute(sqlalchemy.text(
                "INSERT INTO links (linker_uuid, linked_uuid) VALUES ('a', 'b')"))
        engine.dispose()

        with pytest.raises(migrate_links.LinkMigrationError, match='will not drop'):
            migrate_links.migrate(_url(outdated), apply_changes=True, verbose=False)


class TestCommandLine:

    def test_dry_run_then_apply(self, outdated, capsys):
        assert migrate_links.main([outdated.path]) == 0
        assert 'DRY RUN' in capsys.readouterr().out
        assert migrate_links.inspect(_url(outdated))['needs_migration']

        assert migrate_links.main([outdated.path, '--apply']) == 0
        assert not migrate_links.inspect(_url(outdated))['needs_migration']
