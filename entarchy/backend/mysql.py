"""MySQL storage backend.

Everything that is not dialect specific lives in sql.SQLBackend.
"""
import os
from typing import Any
from urllib.parse import quote_plus

import sqlalchemy
from sqlalchemy import create_engine

from .sql import (AttributeTable, Base, EntityTable, EntityTypeTable, Link, SQLBackend,
                  _build_query_from_collection, _deserialize, _generate_attribute_filters,
                  _get_attribute_fp, _get_entity_type_ancestor_distance, _get_namehash,
                  _read_attribute_data, _retry_on_operational_failure, _write_attribute_data)

# The table classes are re-exported because both backends are written against
#  them; nothing here exists for the sake of already-stored data
__all__ = ['MySQLBackend', 'Base', 'EntityTable', 'EntityTypeTable',
           'AttributeTable', 'Link']


class MySQLBackend(SQLBackend):
    """Server backend, for datasets shared between machines or users."""

    def __init__(self, *args, dbname: str, dbhost: str, dbuser: str, dbpassword: str = None,
                 debug: bool = False, **kwargs):
        SQLBackend.__init__(self, *args, **kwargs)

        # Get connection parameters
        if dbhost is None:
            dbhost = input(f'MySQL host name [default: "localhost"]: ')
            if dbhost == '':
                dbhost = 'localhost'

        if dbname is None:
            while True:
                dbname = input(f'New database schema name on host "{dbhost}": ')
                if dbname == '':
                    print('Database schema name cannot be empty.')
                else:
                    break

        if dbuser is None:
            dbuser = input(f'User name for database schema "{dbname}" [default: entarchy_user]: ')
            if dbuser == '':
                dbuser = 'entarchy_user'

        # Passwords are not persisted to entarchy.yaml. Resolution order:
        #  explicit argument (incl. legacy configs) > ENTARCHY_DB_PASSWORD > prompt
        if dbpassword is None:
            dbpassword = os.environ.get('ENTARCHY_DB_PASSWORD')

        if dbpassword is None:
            import getpass
            dbpassword = getpass.getpass(f'Password for user {dbuser} '
                                         f'(set ENTARCHY_DB_PASSWORD to skip this prompt): ')

        self._config = {
            'dbname': dbname,
            'dbhost': dbhost,
            'dbuser': dbuser,
            'dbpassword': dbpassword,
        }

        self.debug = debug

    def get_config(self) -> dict[str, Any]:
        # Persistable configuration: never write the password to entarchy.yaml
        _config = super().get_config()
        _config.pop('dbpassword', None)
        return _config

    @property
    def dbhost(self) -> str:
        return self._config['dbhost']

    @property
    def dbuser(self) -> str:
        return self._config['dbuser']

    @property
    def dbpassword(self) -> str:
        return self._config['dbpassword']

    @property
    def _server_url(self) -> str:
        """Connection URL for the server (without schema). Credentials are URL-escaped."""
        return f'mysql+pymysql://{quote_plus(self.dbuser)}:{quote_plus(self.dbpassword)}@{self.dbhost}'

    def _create_engine(self) -> sqlalchemy.Engine:
        return create_engine(f'{self._server_url}/{self.dbname}',
                             echo=self.debug,
                             pool_size=1,
                             pool_recycle=60,
                             pool_pre_ping=True,
                             )

    @property
    def db_triggers_enabled(self) -> bool:
        if self._db_triggers_enabled is None:
            with self.sql_engine.connect() as conn:
                res = conn.execute(
                    sqlalchemy.text(
                        'SELECT TRIGGER_NAME '
                        'FROM information_schema.TRIGGERS '
                        f'WHERE TRIGGER_SCHEMA = \'{self.dbname}\''
                        'AND TRIGGER_NAME = \'attributes_touch_entities_ai\''
                    )
                )
                self._db_triggers_enabled = len(res.fetchall()) > 0

        return self._db_triggers_enabled

    def _create_database(self) -> None:
        print(f'> Create database {self.dbname}')

        engine = create_engine(self._server_url, echo=self.debug)
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'CREATE SCHEMA IF NOT EXISTS {self.dbname}'))
        engine.dispose()

    def _create_triggers(self) -> None:
        print('> Create triggers')

        with self.sql_engine.connect() as connection:
            try:
                # TODO: this needs to use non-utc datetime
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TRIGGER attributes_touch_entities_ai\n"
                        "AFTER INSERT ON attributes FOR EACH ROW\n"
                        "UPDATE entities\n"
                        "SET modified = UTC_TIMESTAMP(6)\n"
                        "WHERE entities.uuid = NEW.entity_uuid\n"
                    )
                )
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TRIGGER attributes_touch_entities_au\n"
                        "AFTER UPDATE ON attributes FOR EACH ROW\n"
                        "UPDATE entities\n"
                        "SET modified = UTC_TIMESTAMP(6)\n"
                        "WHERE entities.uuid = NEW.entity_uuid\n"
                    )
                )
                connection.execute(
                    sqlalchemy.text(
                        "CREATE TRIGGER attributes_touch_entities_ad\n"
                        "AFTER DELETE ON attributes FOR EACH ROW\n"
                        "UPDATE entities\n"
                        "SET modified = UTC_TIMESTAMP(6)\n"
                        "WHERE entities.uuid = OLD.entity_uuid\n"
                    )
                )
            except:
                # If this fails after successful creation of tables, the most likely reason is detailed here:
                #  https://stackoverflow.com/a/56390000
                print('WARNING: Failed to create database triggers. '
                      'This may impact performance of attribute updates.')
                print(10 * ' ' + 'A likely cause for this are server security settings '
                                 'or insufficient privileges of the database user.')
                print(10 * ' ' + f'For better performance give global \'SUPER\' privilege to dbuser \'{self.dbuser}\' ')
                print(10 * ' ' + f'OR set log-bin-trust-function-creators=1 in MySQL config and restart the server.')

    def _drop_storage(self) -> None:
        print('> Drop schema')

        engine = create_engine(self._server_url)
        with engine.connect() as connection:
            connection.execute(sqlalchemy.text(f'DROP SCHEMA IF EXISTS {self.dbname}'))
            connection.commit()
        engine.dispose()

    def _insert_statement(self, values: list[dict]):
        return sqlalchemy.dialects.mysql.insert(AttributeTable).values(values)

    def _inserted_values(self, insert_statement):
        return insert_statement.inserted

    def _upsert_statement(self, insert_statement, update_values: dict):
        return insert_statement.on_duplicate_key_update(update_values)
