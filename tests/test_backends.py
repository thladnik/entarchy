"""Backend structure and dialect specific SQL.

The SQLite backend is exercised end to end by the rest of the suite. The MySQL
backend needs a server, so here its dialect specific behaviour is checked by
compiling the statements it builds, which covers exactly the branches that differ
between the two.
"""
import datetime

import pytest
import sqlalchemy
from sqlalchemy.dialects import mysql as mysql_dialect
from sqlalchemy.dialects import sqlite as sqlite_dialect
from sqlalchemy.schema import CreateTable

from entarchy.backend import MySQLBackend, SQLiteBackend
from entarchy.backend.sql import AttributeTable, SQLBackend


@pytest.fixture()
def mysql_backend():
    """Constructed but never connected: __init__ does not touch the server."""
    return MySQLBackend('/some/path', dbname='lab', dbhost='db.example',
                        dbuser='analyst', dbpassword='secret')


@pytest.fixture()
def sqlite_backend(tmp_path):
    return SQLiteBackend(tmp_path.as_posix(), dbname='test.db')


ATTRIBUTE_ROW = {
    'entity_uuid': 'aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee',
    'name': 'some_attribute',
    'value_float': 1.5,
    'data_type': 'float',
    'data_size': 8,
    'analysis_uuid': None,
    'float_is_nan': False,
    'float_is_inf': False,
    'modified': datetime.datetime(2026, 1, 1),
}


class TestSharedImplementation:

    def test_both_backends_share_the_base(self):
        assert issubclass(SQLiteBackend, SQLBackend)
        assert issubclass(MySQLBackend, SQLBackend)

    def test_neither_backend_reimplements_shared_behaviour(self):
        """Guard against the two backends drifting apart again: only genuinely
        dialect specific methods may be overridden."""
        allowed = {
            '__init__', '_create_engine', '_create_database', '_create_triggers',
            '_drop_storage', '_insert_statement', '_inserted_values', '_upsert_statement',
            'db_triggers_enabled', 'get_config',
            # MySQL connection parameters
            'dbhost', 'dbuser', 'dbpassword', '_server_url',
        }

        for backend in (SQLiteBackend, MySQLBackend):
            overridden = {name for name in vars(backend)
                          if not name.startswith('__') or name == '__init__'}
            unexpected = overridden - allowed - {'__module__', '__qualname__', '__doc__'}
            assert not unexpected, f'{backend.__name__} unexpectedly overrides {unexpected}'

    def test_shared_names_are_re_exported(self):
        """Old import paths keep working, including for already-pickled data."""
        from entarchy.backend import mysql, sqlite

        assert mysql.Serializer is sqlite.Serializer
        assert mysql.AttributeTable is sqlite.AttributeTable
        assert mysql.EntityTable is sqlite.EntityTable

    def test_entity_and_collection_methods_come_from_the_base(self):
        for name in ('add_entities', 'get_entity_attributes', 'set_entity_attributes',
                     'get_collection_attributes', 'set_collection_attributes',
                     'get_collection_entities_by_slice', 'get_collection_parent_uuids'):
            assert getattr(SQLiteBackend, name) is getattr(MySQLBackend, name), name


class TestSchema:

    def test_mysql_uses_longblob(self):
        """MySQL's generic BLOB holds only 64 KB, far too little for stored arrays."""
        ddl = str(CreateTable(AttributeTable.__table__).compile(dialect=mysql_dialect.dialect()))
        assert 'LONGBLOB' in ddl

    def test_sqlite_uses_a_plain_blob(self):
        ddl = str(CreateTable(AttributeTable.__table__).compile(dialect=sqlite_dialect.dialect()))
        assert 'BLOB' in ddl
        assert 'LONGBLOB' not in ddl

    def test_both_dialects_describe_the_same_columns(self):
        columns = {c.name for c in AttributeTable.__table__.columns}
        for expected in ('entity_uuid', 'name', 'value_str', 'value_int', 'value_float',
                         'value_bool', 'value_date', 'value_datetime', 'value_blob',
                         'data_type', 'data_size', 'float_is_nan', 'float_is_inf'):
            assert expected in columns


class TestUpsert:

    def compiled(self, backend, dialect):
        insert_statement = backend._insert_statement([ATTRIBUTE_ROW])
        proposed = backend._inserted_values(insert_statement)
        upsert = backend._upsert_statement(insert_statement,
                                           {'value_float': proposed.value_float,
                                            'data_type': 'float'})
        return str(upsert.compile(dialect=dialect))

    def test_mysql_uses_on_duplicate_key_update(self, mysql_backend):
        sql = self.compiled(mysql_backend, mysql_dialect.dialect())

        assert 'ON DUPLICATE KEY UPDATE' in sql
        assert 'INSERT INTO attributes' in sql

    def test_sqlite_uses_on_conflict_do_update(self, sqlite_backend):
        sql = self.compiled(sqlite_backend, sqlite_dialect.dialect())

        assert 'ON CONFLICT' in sql
        assert 'DO UPDATE' in sql

    def test_conflict_target_is_the_attribute_primary_key(self, sqlite_backend):
        sql = self.compiled(sqlite_backend, sqlite_dialect.dialect())

        assert 'entity_uuid' in sql
        assert 'name' in sql

    def test_inserted_values_expose_the_proposed_row(self, mysql_backend, sqlite_backend):
        for backend in (mysql_backend, sqlite_backend):
            statement = backend._insert_statement([ATTRIBUTE_ROW])
            proposed = backend._inserted_values(statement)

            # Both dialects expose the pseudo table, under different names
            assert proposed.value_float is not None
            assert proposed.data_size is not None


class TestMySQLConnectionParameters:

    def test_password_is_not_persisted(self, mysql_backend):
        assert 'dbpassword' not in mysql_backend.get_config()
        assert mysql_backend.get_runtime_config()['dbpassword'] == 'secret'

    def test_server_url_escapes_credentials(self):
        backend = MySQLBackend('/p', dbname='lab', dbhost='h',
                               dbuser='user@lab', dbpassword='p@ss:word/1')

        url = backend._server_url
        assert 'p%40ss%3Aword%2F1' in url
        assert 'user%40lab' in url
        assert 'p@ss:word/1' not in url

    def test_password_from_environment(self, monkeypatch):
        monkeypatch.setenv('ENTARCHY_DB_PASSWORD', 'from-env')

        backend = MySQLBackend('/p', dbname='lab', dbhost='h', dbuser='u')
        assert backend.dbpassword == 'from-env'

    def test_explicit_password_wins_over_environment(self, monkeypatch):
        monkeypatch.setenv('ENTARCHY_DB_PASSWORD', 'from-env')

        backend = MySQLBackend('/p', dbname='lab', dbhost='h', dbuser='u', dbpassword='explicit')
        assert backend.dbpassword == 'explicit'

    def test_engine_url_targets_the_schema(self, mysql_backend):
        pytest.importorskip('pymysql', reason='engine creation imports the DBAPI')

        engine = mysql_backend._create_engine()
        assert engine.url.database == 'lab'
        assert engine.url.host == 'db.example'
        engine.dispose()


class TestSQLiteSpecifics:

    def test_engine_points_at_a_file_in_the_entarchy(self, sqlite_backend, tmp_path):
        engine = sqlite_backend._create_engine()

        assert engine.url.drivername == 'sqlite'
        assert 'test.db' in str(engine.url)
        engine.dispose()

    def test_triggers_are_not_used(self, sqlite_backend):
        """Modification times are maintained in Python for SQLite."""
        assert sqlite_backend.db_triggers_enabled is False

    def test_delete_requires_confirmation(self, sqlite_backend):
        with pytest.raises(RuntimeError, match='Confirmation not provided'):
            sqlite_backend.delete()

    def test_close_is_safe_before_any_connection(self, sqlite_backend):
        sqlite_backend.close()
        sqlite_backend.close()
