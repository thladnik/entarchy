"""SQLite storage backend.

Everything that is not dialect specific lives in sql.SQLBackend.
"""
import os
import pathlib
from typing import Any

import sqlalchemy
from sqlalchemy import create_engine

from .sql import (AttributeTable, Base, EntityTable, EntityTypeTable, Link, SQLBackend, Serializer,
                  _build_query_from_collection, _deserialize, _generate_attribute_filters,
                  _get_attribute_fp, _get_entity_type_ancestor_distance, _get_namehash,
                  _read_attribute_data, _retry_on_operational_failure, _write_attribute_data)

# Re-exported so that "from entarchy.backend.sqlite import Serializer" and similar
#  keep working, and so stored pickles referencing this module still resolve
__all__ = ['SQLiteBackend', 'Serializer', 'Base', 'EntityTable', 'EntityTypeTable',
           'AttributeTable', 'Link']


class SQLiteBackend(SQLBackend):
    """Single file backend. Needs no server, so it suits local and single user work."""

    # How many rows PRAGMA optimize may sample per index. Without a limit an
    #  ANALYZE reads every index in full; with one it stays in the hundreds of
    #  milliseconds even on a gigabyte-scale entarchy. 400 is the value SQLite's
    #  own documentation recommends.
    analysis_limit: int = 400

    # An archive opens its index read-only and must not be written to on close
    optimize_on_close: bool = True

    def __init__(self, *args, dbname: str = 'entarchy.db', debug: bool = False, **kwargs):
        SQLBackend.__init__(self, *args, **kwargs)

        self._config = {
            'dbname': dbname,
        }

        self.debug = debug

    def _create_engine(self) -> sqlalchemy.Engine:
        root_path = pathlib.Path(self._root_path).as_posix()
        engine = create_engine(f'sqlite:///{root_path}/{self.dbname}', echo=self.debug,
                               # Wait for concurrent writers instead of failing
                               #  immediately with "database is locked"
                               connect_args={'timeout': 30})

        sqlalchemy.event.listen(engine, 'connect', self._configure_connection)

        return engine

    def _configure_connection(self, dbapi_connection, _record) -> None:
        """Bound the work PRAGMA optimize is allowed to do on this connection."""
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(f'PRAGMA analysis_limit={int(self.analysis_limit)}')
        finally:
            cursor.close()

    def optimize(self) -> None:
        """Let SQLite refresh the query planner statistics it is missing.

        Without a sqlite_stat1 table the planner guesses at how selective an
        index is, and for entarchy's filters it guesses badly: it drives the
        query from the entity type - reading every entity of that type - rather
        than from the attribute the filter names. On 27 000 ROIs a selective
        lookup took 20 ms that way and 0.4 ms once statistics existed.

        PRAGMA optimize only analyses what this session actually touched and
        what has changed enough to matter, so repeat calls cost nothing. Failure
        is not worth reporting: statistics are an optimisation, and a database
        that cannot take them still answers every query correctly.
        """
        if not self.optimize_on_close or self._sql_engine is None:
            return

        try:
            with self._sql_engine.connect() as connection:
                connection.execute(sqlalchemy.text('PRAGMA optimize'))
                connection.commit()
        except Exception:
            pass

    def close(self):
        self.optimize()
        SQLBackend.close(self)

    def _create_database(self) -> None:
        # The file is created by SQLAlchemy on first connect
        print(f'> Create database {self.dbname}')

    def _drop_storage(self) -> None:
        print('> Remove database file')

        db_path = os.path.join(str(self._root_path), self.dbname)
        if os.path.exists(db_path):
            os.remove(db_path)

    # SQLite supports triggers, but the entity modification times are maintained
    #  in Python instead, so db_triggers_enabled stays False (see SQLBackend)

    def _insert_statement(self, values: list[dict]):
        return sqlalchemy.dialects.sqlite.insert(AttributeTable).values(values)

    def _inserted_values(self, insert_statement):
        return insert_statement.excluded

    def _upsert_statement(self, insert_statement, update_values: dict):
        return insert_statement.on_conflict_do_update(
            index_elements=[AttributeTable.entity_uuid, AttributeTable.name],
            set_=update_values,
        )
