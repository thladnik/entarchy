"""Read-only backend for exported ASDF archives.

An archive is an ordinary entarchy directory:

    archive/
        entarchy.yaml     names this backend
        index.sqlite      the queryable metadata, in the normal entarchy schema
        meta.asdf         the same metadata again, self-describing
        blocks/*.asdf     the arrays, one file per parent group

So `Entarchy('/path/to/archive')` opens it and every query, DataFrame and
`map_async` call works unchanged - the analysis and figure code that reads a
live entarchy reads an archive without modification.

The split is deliberate. Queries run against index.sqlite, which is a plain
SQLite database and therefore has indexes and a query planner; ASDF has
neither, and its YAML tree has to be parsed in full on open, measured at
roughly 0.4 ms per entry even with lazy_tree. Only array reads touch ASDF,
where the cost is flat regardless of file size.

index.sqlite is a derived cache. meta.asdf holds everything needed to rebuild
it (`python -m entarchy.tools.archive rebuild <archive>`), which is what
keeps the archival claim honest: the durable, self-describing copy is the ASDF,
and the database is an accelerator that can always be regenerated.

Writes are rejected. An archive is a fixed record of a dataset; to continue
working on one, import it back into a normal entarchy.
"""
from __future__ import annotations

import os
import pathlib
from typing import Any

import sqlalchemy
from sqlalchemy import create_engine

from . import asdf_store
from .sql import AttributeTable, Base, EntityTable, EntityTypeTable, Link
from .sqlite import SQLiteBackend

__all__ = ['ArchiveBackend', 'ArchiveReadOnlyError', 'Base',
           'EntityTable', 'EntityTypeTable', 'AttributeTable', 'Link']

INDEX_NAME = 'index.sqlite'
META_NAME = 'meta.asdf'
BLOCK_DIR = 'blocks'


class ArchiveReadOnlyError(RuntimeError):
    """Raised when something tries to modify an archive."""


class ArchiveBackend(SQLiteBackend):
    """Opens an exported archive. Reads like SQLite, refuses to write."""

    # The index is opened read-only, and an archive is meant to be a fixed
    #  artefact - refreshing planner statistics would write to it. The export
    #  analyses the index once instead, so archives ship with statistics already.
    optimize_on_close = False

    def __init__(self, *args, dbname: str = INDEX_NAME, memmap: bool = False,
                 open_file_limit: int = None, debug: bool = False, **kwargs):
        SQLiteBackend.__init__(self, *args, dbname=dbname, debug=debug, **kwargs)

        self._config = {
            'dbname': dbname,
            # Memory mapped arrays stop being readable once their file is closed,
            #  and block files are closed when the open-file cache evicts them, so
            #  this is off unless a caller knows the archive stays open
            'memmap': memmap,
        }

        # Raise this on an archive with many parent groups: a limit below the
        #  number of groups a pass touches makes every read evict a file the next
        #  read wants, which measured 13x slower than holding them all open
        if open_file_limit is not None:
            self._config['open_file_limit'] = open_file_limit
            asdf_store.set_open_file_limit(open_file_limit)

        self.debug = debug

    @property
    def memmap(self) -> bool:
        return self._config.get('memmap', False)

    def _create_engine(self) -> sqlalchemy.Engine:
        root_path = pathlib.Path(self._root_path).as_posix()
        index_path = f'{root_path}/{self.dbname}'

        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f'Archive at "{root_path}" has no {self.dbname}. If the archive was '
                f'deposited without it, rebuild it from {META_NAME} with:\n'
                f'    python -m entarchy.tools.archive rebuild "{root_path}"')

        # Ask SQLite itself to refuse writes, so a bug here cannot damage an archive.
        #  The URI form is not accepted everywhere, so a plain connection is the
        #  fallback; the method guards below still apply either way.
        try:
            engine = create_engine(f'sqlite:///file:{index_path}?mode=ro&uri=true',
                                   echo=self.debug)
            with engine.connect() as connection:
                connection.execute(sqlalchemy.text('SELECT 1'))
            return engine
        except Exception:
            return create_engine(f'sqlite:///{index_path}', echo=self.debug)

    def close(self):
        SQLiteBackend.close(self)
        asdf_store.close_asdf_files()

    # Everything that would modify the archive

    def _refuse(self, what: str):
        raise ArchiveReadOnlyError(
            f'Cannot {what}: "{self._root_path}" is an exported archive and is read-only. '
            f'To keep working with this data, import it into a normal entarchy:\n'
            f'    python -m entarchy.tools.archive import "{self._root_path}" <destination>')

    def create(self) -> bool:
        self._refuse('create storage')

    def create_type_hierarchy(self, _hierarchy: dict[str, Any]) -> bool:
        self._refuse('create the type hierarchy')

    def delete(self, confirm: bool = False):
        self._refuse('delete storage')

    def add_entity(self, _entity) -> bool:
        self._refuse('add an entity')

    def add_entities(self, _entities) -> bool:
        self._refuse('add entities')

    def set_entity_attribute(self, _entity, name: str, value: Any):
        self._refuse(f'set attribute "{name}"')

    def set_entity_attributes(self, _entity, names, value):
        self._refuse('set attributes')

    def set_collection_attributes(self, _collection, df):
        self._refuse('set collection attributes')
