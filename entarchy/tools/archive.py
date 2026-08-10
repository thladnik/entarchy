"""Export an entarchy to a self-describing ASDF archive, and read it back.

An archive is a normal entarchy directory that happens to keep its arrays in
ASDF and its metadata in both a SQLite index and a self-describing ASDF file:

    archive/
        entarchy.yaml     names entarchy.backend.archive.ArchiveBackend
        index.sqlite      queryable metadata, normal entarchy schema
        meta.asdf         the same metadata, columnar and self-describing
        blocks/*.asdf     the arrays, one file per parent group

`Entarchy('/path/to/archive')` opens it, so existing analysis and figure code
reads an archive without modification.

Three commands:

    python -m entarchy.tools.archive export <source> <archive> [--query ...]
    python -m entarchy.tools.archive rebuild <archive>
    python -m entarchy.tools.archive import <archive> <destination>

`export` writes the archive. `rebuild` regenerates index.sqlite from meta.asdf,
which is what makes the index a cache rather than a second source of truth.
`import` turns an archive back into an ordinary SQLite entarchy that can be
written to again.

Metadata is stored columnar: one array per column rather than one YAML node per
row. A tree with a node per attribute would have to be parsed in full on open,
measured at roughly 0.4 ms per entry, which for a dataset of 100 000 entities
would be minutes. As arrays it is a handful of binary blocks.
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import re
import shutil
import sys
from typing import Any, Union

import numpy as np
import sqlalchemy
import yaml
from sqlalchemy.orm import Session

from ..backend import asdf_store, blob_store
from ..backend.archive import BLOCK_DIR, INDEX_NAME, META_NAME
from ..backend.sql import (AttributeTable, Base, EntityTable, EntityTypeTable, Link,
                          LinkTypeTable, _store_blob)

ARCHIVE_FORMAT_VERSION = 1

# Column groups copied verbatim between the index and meta.asdf
_ENTITY_COLUMNS = ['uuid', 'parent_uuid', 'entity_type_pk', 'id', 'created', 'modified']
_LINK_COLUMNS = ['link_uuid', 'link_type', 'linker_uuid', 'linked_uuid', 'created', 'modified']

# Without these an archive would carry links whose meaning had been lost
_LINK_TYPE_COLUMNS = ['name', 'linker_type_pk', 'linker_link_type', 'linked_type_pk',
                      'linked_link_type', 'symmetric', 'cardinality', 'description',
                      'created']
_ATTRIBUTE_COLUMNS = ['entity_uuid', 'analysis_uuid', 'name', 'value_str', 'value_int',
                      'value_float', 'value_bool', 'value_date', 'value_datetime',
                      'data_type', 'data_size', 'float_is_nan', 'float_is_inf', 'mutable',
                      'created', 'modified']

# How each column is represented in meta.asdf. Every column also gets a boolean
#  null mask, because None and a legitimate zero/empty string/NaN must survive
#  the round trip distinctly.
_COLUMN_KINDS = {
    'uuid': 'str', 'parent_uuid': 'str', 'entity_uuid': 'str', 'analysis_uuid': 'str',
    'linker_uuid': 'str', 'linked_uuid': 'str', 'id': 'str', 'name': 'str',
    'value_str': 'str', 'data_type': 'str',
    'link_uuid': 'str', 'link_type': 'str', 'linker_link_type': 'str',
    'linked_link_type': 'str', 'cardinality': 'str', 'description': 'str',
    'entity_type_pk': 'int', 'value_int': 'int', 'data_size': 'int',
    'linker_type_pk': 'int', 'linked_type_pk': 'int',
    'value_float': 'float',
    'value_bool': 'bool', 'float_is_nan': 'bool', 'float_is_inf': 'bool', 'mutable': 'bool',
    'symmetric': 'bool',
    'created': 'datetime', 'modified': 'datetime', 'value_datetime': 'datetime',
    'value_date': 'date',
}


class ExportError(RuntimeError):
    pass


def _sanitize(name: str) -> str:
    """Make an entity id usable as a file name."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(name)).strip('_')
    return cleaned[:60] if len(cleaned) > 0 else 'group'


def _to_column(values: list, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Turn a column of Python values into an array plus its null mask."""
    mask = np.array([value is None for value in values], dtype=bool)

    if kind == 'str':
        return np.array(['' if value is None else str(value) for value in values], dtype=np.str_), mask
    if kind == 'int':
        return np.array([0 if value is None else int(value) for value in values], dtype=np.int64), mask
    if kind == 'float':
        return np.array([0.0 if value is None else float(value) for value in values], dtype=np.float64), mask
    if kind == 'bool':
        return np.array([False if value is None else bool(value) for value in values], dtype=bool), mask
    if kind in ('datetime', 'date'):
        return np.array(['' if value is None else value.isoformat() for value in values],
                        dtype=np.str_), mask

    raise ValueError(f'Unknown column kind "{kind}"')


def _from_column(data: np.ndarray, mask: np.ndarray, kind: str) -> list:
    """Inverse of _to_column."""
    values = []
    for index in range(len(data)):
        if mask[index]:
            values.append(None)
            continue

        raw = data[index]
        if kind == 'str':
            values.append(str(raw))
        elif kind == 'int':
            values.append(int(raw))
        elif kind == 'float':
            values.append(float(raw))
        elif kind == 'bool':
            values.append(bool(raw))
        elif kind == 'datetime':
            values.append(datetime.datetime.fromisoformat(str(raw)))
        elif kind == 'date':
            values.append(datetime.date.fromisoformat(str(raw)))
        else:
            raise ValueError(f'Unknown column kind "{kind}"')

    return values


def _table_to_tree(rows: list[dict], columns: list[str]) -> dict:
    tree = {'count': len(rows)}
    for column in columns:
        data, mask = _to_column([row[column] for row in rows], _COLUMN_KINDS[column])
        tree[column] = data
        tree[f'{column}__null'] = mask

    return tree


def _tree_to_table(tree: dict, columns: list[str]) -> list[dict]:
    count = int(tree['count'])
    if count == 0:
        return []

    decoded = {column: _from_column(np.asarray(tree[column]),
                                    np.asarray(tree[f'{column}__null']),
                                    _COLUMN_KINDS[column])
               for column in columns}

    return [{column: decoded[column][index] for column in columns} for index in range(count)]


class _EntarchyHandle:
    """Enough of an Entarchy to read rows and deserialize blobs.

    The concrete Entarchy subclass defines the entity hierarchy, and opening a
    directory through `Entarchy(path)` validates that hierarchy against the
    classes the caller has imported. Archiving does not need any of that - it
    copies rows and payloads - and requiring it would mean the command line tool
    could only run where the schema package happens to be importable. So the
    backend is built straight from entarchy.yaml instead.
    """

    def __init__(self, path: str):
        import importlib

        self.path = pathlib.Path(path).absolute().as_posix()

        config_path = os.path.join(self.path, 'entarchy.yaml')
        if not os.path.exists(config_path):
            raise ExportError(f'"{self.path}" has no entarchy.yaml; it is not an entarchy.')

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        parts = self.config['backend'].split('.')
        backend_cls = getattr(importlib.import_module('.'.join(parts[:-1])), parts[-1])
        self.backend = backend_cls(self.path, **self.config.get('backend_config', {}),
                                   debug=False)

    @property
    def max_blob_size(self) -> int:
        return self.config.get('max_blob_size', 0)


def _open_source(source: Union[str, Any]):
    """Accept an Entarchy instance or a path to one."""
    if isinstance(source, (str, os.PathLike, pathlib.Path)):
        return _EntarchyHandle(str(source)), True

    return source, False


def _select_uuids(source, collection) -> Union[set, None]:
    """The entity uuids to export, or None for everything.

    A collection selects its own entities plus every ancestor, so that parent
    lookups and `[Parent]attr` filters still resolve inside the archive.
    """
    if collection is None:
        return None

    with Session(source.backend.sql_engine) as session:
        parents = dict(session.query(EntityTable.uuid, EntityTable.parent_uuid).all())

    selected = {uuid for uuid, _ in source.backend.get_collection_parent_uuids(collection)}

    # Walk up to the root so no exported entity has a dangling parent
    pending = list(selected)
    while len(pending) > 0:
        uuid = pending.pop()
        parent = parents.get(uuid)
        if parent is not None and parent not in selected:
            selected.add(parent)
            pending.append(parent)

    return selected


# Metadata columns are fixed width numpy strings, so every row is padded to the
#  longest value in its column - 36 bytes of uuid become 144 in UCS4, and one long
#  value_str widens the whole column. That padding is pure redundancy and compresses
#  away almost entirely: 2185 kB to 65 kB on a 1994 attribute export. meta.asdf is
#  only read when the index is rebuilt, so the cost of decompressing it is rare.
#  Array blocks are left alone by default: imaging traces compress poorly and are
#  read on the hot path.
META_COMPRESSION = 'zlib'


def export(source: Union[str, Any],
           destination: str,
           collection=None,
           compression: str = None,
           meta_compression: str = META_COMPRESSION,
           skip_broken: bool = False,
           overwrite: bool = False,
           verbose: bool = True) -> dict:
    """Write an entarchy, or a collection out of one, to an ASDF archive."""
    asdf_store.require()
    import asdf

    source_ent, opened_here = _open_source(source)

    try:
        return _export(source_ent, destination, collection=collection, compression=compression,
                       meta_compression=meta_compression, skip_broken=skip_broken,
                       overwrite=overwrite, verbose=verbose, asdf_module=asdf)
    finally:
        if opened_here:
            source_ent.backend.close()


def _export(source_ent, destination, collection, compression, meta_compression, skip_broken,
            overwrite, verbose, asdf_module):
    destination = str(pathlib.Path(destination).absolute())

    if os.path.exists(destination):
        if not overwrite:
            raise ExportError(f'"{destination}" already exists. Pass overwrite=True (--overwrite) '
                              f'to replace it.')
        shutil.rmtree(destination)

    os.makedirs(os.path.join(destination, BLOCK_DIR), exist_ok=False)

    selected = _select_uuids(source_ent, collection)
    report = asdf_store.EncodingReport()
    stats = {'entities': 0, 'attributes': 0, 'blobs': 0, 'block_files': 0,
             'broken': [], 'bytes': 0}

    index_engine = sqlalchemy.create_engine(
        f'sqlite:///{pathlib.Path(destination).as_posix()}/{INDEX_NAME}')
    Base.metadata.create_all(index_engine)

    with Session(source_ent.backend.sql_engine) as session:
        # Entity types are few and always copied whole, so an archive keeps the
        #  complete hierarchy even when only a subtree of entities is exported
        type_rows = [{'pk': row.pk, 'parent_pk': row.parent_pk, 'name': row.name}
                     for row in session.query(EntityTypeTable).all()]

        entity_query = session.query(EntityTable)
        if selected is not None:
            entity_query = entity_query.filter(EntityTable.uuid.in_(list(selected)))
        entity_rows = [{column: getattr(row, column) for column in _ENTITY_COLUMNS}
                       for row in entity_query.all()]

        uuids = {row['uuid'] for row in entity_rows}
        stats['entities'] = len(entity_rows)

        # Link kinds are copied whole, so an archive explains its links even when
        #  only a subtree of entities was exported
        link_type_rows = [{column: getattr(row, column) for column in _LINK_TYPE_COLUMNS}
                          for row in session.query(LinkTypeTable).all()]

        link_rows = [{column: getattr(row, column) for column in _LINK_COLUMNS}
                     for row in session.query(Link).all()
                     if row.linker_uuid in uuids and row.linked_uuid in uuids
                     and row.link_uuid in uuids]

        # Group entities by parent. Block files follow the same grouping map_async
        #  uses for worker locality, so reading a recording touches one file.
        entity_by_uuid = {row['uuid']: row for row in entity_rows}
        groups: dict[Union[str, None], list[str]] = {}
        for row in entity_rows:
            groups.setdefault(row['parent_uuid'], []).append(row['uuid'])

        type_name_by_pk = {row['pk']: row['name'] for row in type_rows}

        attribute_rows: list[dict] = []
        for group_index, (parent_uuid, member_uuids) in enumerate(sorted(
                groups.items(), key=lambda item: (item[0] is not None, item[0] or ''))):

            group_rows, blobs, relative_path, content_types = _export_group(
                session, source_ent, member_uuids, parent_uuid, entity_by_uuid,
                type_name_by_pk, group_index, report, stats, skip_broken)

            if len(blobs) > 0:
                full_path = os.path.join(destination, relative_path)

                tree = {'entarchy_archive': {'version': ARCHIVE_FORMAT_VERSION,
                                             'encoding_version': asdf_store.ENCODING_VERSION,
                                             'entity_types': sorted(content_types),
                                             'parent_uuid': parent_uuid or '',
                                             'parent_id': (entity_by_uuid[parent_uuid]['id']
                                                           if parent_uuid in entity_by_uuid else ''),
                                             'entity_count': len(member_uuids)},
                        'blobs': blobs}
                _write_asdf(asdf_module, tree, full_path, compression)

                stats['block_files'] += 1
                stats['bytes'] += os.path.getsize(full_path)

                if verbose:
                    print(f'  {relative_path}  {len(blobs)} blob(s), '
                          f'{os.path.getsize(full_path) / 1024 ** 2:.1f} MB')

            attribute_rows.extend(group_rows)

        stats['attributes'] = len(attribute_rows)

    # Write the index
    with index_engine.begin() as connection:
        if len(type_rows) > 0:
            connection.execute(sqlalchemy.insert(EntityTypeTable), type_rows)
        if len(link_type_rows) > 0:
            connection.execute(sqlalchemy.insert(LinkTypeTable), link_type_rows)
        if len(entity_rows) > 0:
            connection.execute(sqlalchemy.insert(EntityTable), entity_rows)
        if len(link_rows) > 0:
            connection.execute(sqlalchemy.insert(Link), link_rows)
        if len(attribute_rows) > 0:
            connection.execute(sqlalchemy.insert(AttributeTable), attribute_rows)

        # An archive is read-only, so this is the only chance to give its index
        #  the statistics SQLite's planner needs. Without them a filter is planned
        #  from the entity type rather than from the attribute it names, which on
        #  27 000 entities is the difference between 20 ms and 0.4 ms.
        connection.execute(sqlalchemy.text('ANALYZE'))
    index_engine.dispose()

    # Write the self-describing copy of the metadata
    meta_tree = {
        'entarchy_archive': {
            'version': ARCHIVE_FORMAT_VERSION,
            'encoding_version': asdf_store.ENCODING_VERSION,
            'exported': datetime.datetime.now().isoformat(),
            'source': str(source_ent.path),
            'note': 'index.sqlite is derived from this file and can be regenerated with '
                    '"python -m entarchy.tools.archive rebuild <archive>".',
        },
        'entity_types': {'count': len(type_rows),
                         'pk': np.array([row['pk'] for row in type_rows], dtype=np.int64),
                         'parent_pk': np.array([-1 if row['parent_pk'] is None else row['parent_pk']
                                                for row in type_rows], dtype=np.int64),
                         'name': np.array([row['name'] for row in type_rows], dtype=np.str_)},
        'entities': _table_to_tree(entity_rows, _ENTITY_COLUMNS),
        'link_types': _table_to_tree(link_type_rows, _LINK_TYPE_COLUMNS),
        'links': _table_to_tree(link_rows, _LINK_COLUMNS),
        'attributes': _table_to_tree(attribute_rows, _ATTRIBUTE_COLUMNS),
        'attribute_blobs': _blob_pointer_tree(attribute_rows),
    }
    meta_path = os.path.join(destination, META_NAME)
    _write_asdf(asdf_module, meta_tree, meta_path, meta_compression)
    stats['bytes'] += os.path.getsize(meta_path)

    _write_config(source_ent, destination)

    stats['report'] = report
    if verbose:
        _print_summary(destination, stats, report)

    return stats


def _write_asdf(asdf, tree: dict, path: str, compression: str = None) -> None:
    handle = asdf.AsdfFile(tree)
    try:
        if compression is not None:
            handle.write_to(path, all_array_compression=compression)
        else:
            handle.write_to(path)
    finally:
        handle.close()


def _group_file_name(parent_uuid, parent_id: str, content_types: set, group_index: int) -> str:
    """Name a block file after what it holds, not after the parent it hangs from.

    A group is the children of one entity, so naming it after the parent reads as
    if it held the parent's own data - "animal_01.asdf" actually holding a
    recording's arrays. The type of the contents goes first, with the parent kept
    as context, because that is the question being asked when someone opens the
    directory: which file has the ROIs of plane0?
    """
    types = _sanitize('-'.join(sorted(content_types))) if content_types else 'entities'

    if parent_uuid is None:
        return f'{BLOCK_DIR}/{group_index:04d}_{types}_at_root.asdf'

    # The uuid fragment keeps names unique when two parents share an id
    return (f'{BLOCK_DIR}/{group_index:04d}_{types}_in_{_sanitize(parent_id)}'
            f'_{str(parent_uuid)[:8]}.asdf')


def _export_group(session, source_ent, member_uuids, parent_uuid, entity_by_uuid,
                  type_name_by_pk, group_index, report, stats,
                  skip_broken) -> tuple[list[dict], dict, str, set]:
    """Copy one group's attribute rows, moving blob payloads into ASDF."""
    rows = session.query(AttributeTable).filter(
        AttributeTable.entity_uuid.in_(list(member_uuids))).all()

    # Name the file after the entities that actually contribute payloads, which is
    #  needed before the first blob pointer is written
    content_types = set()
    for row in rows:
        if row.data_type == 'blob' and row.value_blob is not None:
            entity = entity_by_uuid.get(row.entity_uuid)
            if entity is not None:
                content_types.add(type_name_by_pk.get(entity['entity_type_pk'], 'Entity'))

    parent = entity_by_uuid.get(parent_uuid)
    relative_path = _group_file_name(parent_uuid, parent['id'] if parent else 'group',
                                     content_types, group_index)

    exported: list[dict] = []
    blobs: dict[str, Any] = {}

    for row in rows:
        record = {column: getattr(row, column) for column in _ATTRIBUTE_COLUMNS}
        record['value_blob'] = row.value_blob

        if row.data_type == 'blob' and row.value_blob is not None:
            key = f'{row.entity_uuid}/{row.name}'
            where = f'{entity_by_uuid.get(row.entity_uuid, {}).get("id", "?")}.{row.name}'

            try:
                value = blob_store.loads(row.value_blob, root_path=source_ent.path)
            except Exception as err:
                stats['broken'].append((where, str(err)))
                if not skip_broken:
                    raise ExportError(
                        f'Could not read blob attribute "{row.name}" of entity '
                        f'"{where}": {err}\n'
                        f'Pass --skip-broken to leave the attribute out of the '
                        f'archive.') from err
                continue

            blobs[key] = blob_store.encode(value, report, where)
            # The archive keeps a pointer in exactly the same place a live
            #  entarchy does, which is why nothing above the backend changes
            record['value_blob'] = blob_store.dumps_archived(relative_path, key)
            stats['blobs'] += 1

        exported.append(record)

    return exported, blobs, relative_path, content_types


def _blob_pointer_tree(attribute_rows: list[dict]) -> dict:
    """Where each blob lives, so meta.asdf alone is enough to rebuild the index."""
    keys, stores = [], []
    for row in attribute_rows:
        if row['data_type'] != 'blob' or row.get('value_blob') is None:
            continue

        keys.append(f'{row["entity_uuid"]}/{row["name"]}')
        stores.append(blob_store.store_of(row['value_blob']))

    return {'count': len(keys),
            'key': np.array(keys, dtype=np.str_),
            'store': np.array(stores, dtype=np.str_)}


def _write_config(source_ent, destination: str) -> None:
    """Copy entarchy.yaml, pointing it at the archive backend."""
    with open(os.path.join(source_ent.path, 'entarchy.yaml'), 'r') as f:
        config = yaml.safe_load(f)

    config['backend'] = 'entarchy.backend.archive.ArchiveBackend'
    config['backend_config'] = {'dbname': INDEX_NAME, 'memmap': False}

    with open(os.path.join(destination, 'entarchy.yaml'), 'w') as f:
        yaml.safe_dump(config, f)


def _print_summary(destination: str, stats: dict, report) -> None:
    print()
    print(f'Archive written to {destination}')
    print(f'  {stats["entities"]} entities, {stats["attributes"]} attributes, '
          f'{stats["blobs"]} blobs in {stats["block_files"]} block file(s)')
    print(f'  {stats["bytes"] / 1024 ** 2:.1f} MB total')

    if report.ragged_packed > 0:
        print(f'  {report.ragged_packed} ragged array collection(s) packed into single blocks')

    if not report.is_fully_portable:
        print()
        print(f'  WARNING: {report.summary()}')
        print('  Those values can only be read where the defining Python classes are '
              'importable,')
        print('  which limits how self-describing this archive is. The affected attributes:')
        for where, type_name in report.pickled[:20]:
            print(f'    {where}  ({type_name})')
        if len(report.pickled) > 20:
            print(f'    ... and {len(report.pickled) - 20} more')

    if len(stats['broken']) > 0:
        print()
        print(f'  {len(stats["broken"])} attribute(s) could not be read and were skipped:')
        for where, err in stats['broken'][:10]:
            print(f'    {where}: {err}')


# Rebuilding the index from meta.asdf


def rebuild_index(archive_path: str, verbose: bool = True) -> int:
    """Regenerate index.sqlite from meta.asdf. Returns the number of rows written."""
    asdf_store.require()
    import asdf

    archive_path = str(pathlib.Path(archive_path).absolute())
    meta_path = os.path.join(archive_path, META_NAME)

    if not os.path.exists(meta_path):
        raise ExportError(f'"{archive_path}" has no {META_NAME}; it is not an archive.')

    index_path = os.path.join(archive_path, INDEX_NAME)
    if os.path.exists(index_path):
        os.remove(index_path)

    with asdf.open(meta_path, mode='r', lazy_load=False, memmap=False) as handle:
        type_tree = handle['entity_types']
        type_rows = [{'pk': int(pk),
                      'parent_pk': None if int(parent) < 0 else int(parent),
                      'name': str(name)}
                     for pk, parent, name in zip(type_tree['pk'], type_tree['parent_pk'],
                                                 type_tree['name'])]

        entity_rows = _tree_to_table(handle['entities'], _ENTITY_COLUMNS)
        # Archives written before link types existed have no such entry
        link_type_rows = (_tree_to_table(handle['link_types'], _LINK_TYPE_COLUMNS)
                          if 'link_types' in handle else [])
        link_rows = _tree_to_table(handle['links'], _LINK_COLUMNS)
        attribute_rows = _tree_to_table(handle['attributes'], _ATTRIBUTE_COLUMNS)

        # Blob pointers are held separately, since value_blob holds a pointer
        #  rather than the value
        pointer_tree = handle['attribute_blobs']
        pointers = {}
        for index in range(int(pointer_tree['count'])):
            pointers[str(pointer_tree['key'][index])] = str(pointer_tree['store'][index])

    for row in attribute_rows:
        key = f'{row["entity_uuid"]}/{row["name"]}'
        if row['data_type'] == 'blob' and key in pointers:
            relative_path, blob_key = asdf_store.parse_store(pointers[key])
            row['value_blob'] = blob_store.dumps_archived(relative_path, blob_key)
        else:
            row['value_blob'] = None

    engine = sqlalchemy.create_engine(f'sqlite:///{pathlib.Path(index_path).as_posix()}')
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        if len(type_rows) > 0:
            connection.execute(sqlalchemy.insert(EntityTypeTable), type_rows)
        if len(link_type_rows) > 0:
            connection.execute(sqlalchemy.insert(LinkTypeTable), link_type_rows)
        if len(entity_rows) > 0:
            connection.execute(sqlalchemy.insert(EntityTable), entity_rows)
        if len(link_rows) > 0:
            connection.execute(sqlalchemy.insert(Link), link_rows)
        if len(attribute_rows) > 0:
            connection.execute(sqlalchemy.insert(AttributeTable), attribute_rows)

        # As in export(): a rebuilt index is read-only from here on, so its
        #  planner statistics have to be collected now or never
        connection.execute(sqlalchemy.text('ANALYZE'))
    engine.dispose()

    total = len(type_rows) + len(entity_rows) + len(link_rows) + len(attribute_rows)
    if verbose:
        print(f'Rebuilt {INDEX_NAME}: {len(entity_rows)} entities, '
              f'{len(attribute_rows)} attributes, {len(link_rows)} links')

    return total


# Importing an archive back into a writable entarchy


def import_archive(archive_path: str, destination: str, overwrite: bool = False,
                   verbose: bool = True) -> dict:
    """Turn an archive into an ordinary SQLite entarchy that can be written to."""
    asdf_store.require()

    archive_path = str(pathlib.Path(archive_path).absolute())
    destination = str(pathlib.Path(destination).absolute())

    if os.path.exists(destination):
        if not overwrite:
            raise ExportError(f'"{destination}" already exists. Pass overwrite=True (--overwrite) '
                              f'to replace it.')
        shutil.rmtree(destination)

    os.makedirs(destination, exist_ok=False)

    # The index already has the right schema, so it becomes the new database
    shutil.copyfile(os.path.join(archive_path, INDEX_NAME),
                    os.path.join(destination, 'entarchy.db'))

    with open(os.path.join(archive_path, 'entarchy.yaml'), 'r') as f:
        config = yaml.safe_load(f)
    config['backend'] = 'entarchy.backend.sqlite.SQLiteBackend'
    config['backend_config'] = {'dbname': 'entarchy.db'}
    with open(os.path.join(destination, 'entarchy.yaml'), 'w') as f:
        yaml.safe_dump(config, f)

    # Move the payloads out of ASDF and into the ext/ layout a live entarchy uses
    archive_ent = _EntarchyHandle(archive_path)
    stats = {'blobs': 0}

    try:
        engine = sqlalchemy.create_engine(
            f'sqlite:///{pathlib.Path(destination).as_posix()}/entarchy.db')
        with Session(engine) as session:
            rows = session.query(AttributeTable).filter(
                AttributeTable.data_type == 'blob').all()

            for row in rows:
                if row.value_blob is None:
                    continue

                if not blob_store.store_of(row.value_blob).startswith(asdf_store.STORE_PREFIX):
                    continue

                value = blob_store.loads(row.value_blob, root_path=archive_ent.path)

                # max_blob_size 0 puts every payload in a file of its own, which
                #  is what the live ext/ layout is
                row.value_blob = _store_blob(value, destination, 0,
                                             row.entity_uuid, row.name)
                stats['blobs'] += 1

            session.commit()
        engine.dispose()
    finally:
        archive_ent.backend.close()

    if verbose:
        print(f'Imported {archive_path} to {destination} ({stats["blobs"]} blob(s) materialised)')

    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m entarchy.tools.archive',
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest='command', required=True)

    export_parser = subparsers.add_parser('export', help='write an entarchy to an ASDF archive')
    export_parser.add_argument('source', help='entarchy directory to export')
    export_parser.add_argument('destination', help='archive directory to create')
    export_parser.add_argument('--entarchy-class', dest='entarchy_class',
                               help='dotted path to the Entarchy subclass defining the schema, '
                                    'e.g. mypackage.schema.MyArchy. Only needed to export a '
                                    'subset with --type; a full export needs no schema import.')
    export_parser.add_argument('--type', help='entity type to select, e.g. Roi '
                                              '(requires --entarchy-class)')
    export_parser.add_argument('--query', help='filter expression, used with --type')
    export_parser.add_argument('--compression',
                               help='compression for the array blocks, e.g. zlib or lz4. Off by '
                                    'default: imaging traces compress poorly and are read often. '
                                    f'The metadata is always compressed ({META_COMPRESSION}).')
    export_parser.add_argument('--skip-broken', action='store_true',
                               help='leave unreadable blob attributes out instead of failing')
    export_parser.add_argument('--overwrite', action='store_true',
                               help='replace the destination if it exists')

    rebuild_parser = subparsers.add_parser('rebuild', help='regenerate index.sqlite from meta.asdf')
    rebuild_parser.add_argument('archive', help='archive directory')

    import_parser = subparsers.add_parser('import', help='turn an archive back into an entarchy')
    import_parser.add_argument('archive', help='archive directory')
    import_parser.add_argument('destination', help='entarchy directory to create')
    import_parser.add_argument('--overwrite', action='store_true',
                               help='replace the destination if it exists')

    args = parser.parse_args(argv)

    if args.command == 'export':
        if args.type is None:
            export(args.source, args.destination, compression=args.compression,
                   skip_broken=args.skip_broken, overwrite=args.overwrite)
        else:
            if args.entarchy_class is None:
                raise SystemExit('--type needs --entarchy-class, since the entity classes live '
                                 'in your schema package rather than in entarchy.')

            import importlib

            parts = args.entarchy_class.split('.')
            entarchy_cls = getattr(importlib.import_module('.'.join(parts[:-1])), parts[-1])

            source_ent = entarchy_cls(args.source)
            entity_type = source_ent._entity_map.get(args.type)
            if entity_type is None:
                available = ', '.join(sorted(source_ent._entity_map))
                raise SystemExit(f'Unknown entity type "{args.type}". Available: {available}')

            collection = source_ent.get(entity_type, args.query) if args.query \
                else source_ent.get(entity_type)

            try:
                export(source_ent, args.destination, collection=collection,
                       compression=args.compression, skip_broken=args.skip_broken,
                       overwrite=args.overwrite)
            finally:
                source_ent.backend.close()

    elif args.command == 'rebuild':
        rebuild_index(args.archive)

    elif args.command == 'import':
        import_archive(args.archive, args.destination, overwrite=args.overwrite)

    return 0


if __name__ == '__main__':
    sys.exit(main())
