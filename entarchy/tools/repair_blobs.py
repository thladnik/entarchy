"""Repair blob attributes written by the defective collection write path.

Background: set_collection_attributes used to store a pickled pandas Series
(the whole insert row) in value_blob instead of the pickled Serializer object,
which makes those attributes unreadable (AttributeError: 'Series' object has
no attribute 'deserialize'). The original payload is fully recoverable, because
the correct Serializer object is embedded in the stored Series under the
'__serializer' key.

Usage:
    python -m entarchy.tools.repair_blobs <target> [--apply] [--fix-sizes]

<target> may be:
    - the path of an entarchy directory (containing entarchy.yaml)
    - the path of a SQLite database file
    - a full SQLAlchemy URL (e.g. mysql+pymysql://user@host/schema;
      password is taken from ENTARCHY_DB_PASSWORD or prompted)

Without --apply the tool only reports what it would change (dry run).
"""
from __future__ import annotations

import argparse
import datetime
import os
import pathlib
import pickle
import sys

import pandas as pd
import sqlalchemy
import yaml


def _resolve_url(target: str) -> str:
    """Resolve a CLI target (directory, sqlite file or URL) into a SQLAlchemy URL."""

    path = pathlib.Path(target)

    # entarchy directory
    if path.is_dir():
        config_path = path / 'entarchy.yaml'
        if not config_path.exists():
            raise SystemExit(f'{target} is a directory but contains no entarchy.yaml')

        config = yaml.safe_load(open(config_path, 'r'))
        backend = config.get('backend', '')
        backend_config = config.get('backend_config', {})

        if 'sqlite' in backend.lower():
            db_path = (path / backend_config.get('dbname', 'entarchy.db')).as_posix()
            return f'sqlite:///{db_path}'

        if 'mysql' in backend.lower():
            from urllib.parse import quote_plus

            password = backend_config.get('dbpassword') or os.environ.get('ENTARCHY_DB_PASSWORD')
            if password is None:
                import getpass
                password = getpass.getpass(f'Password for {backend_config.get("dbuser")}: ')

            return (f'mysql+pymysql://{quote_plus(backend_config["dbuser"])}:{quote_plus(password)}'
                    f'@{backend_config["dbhost"]}/{backend_config["dbname"]}')

        raise SystemExit(f'Unknown backend {backend!r} in {config_path}')

    # sqlite database file
    if path.is_file():
        return f'sqlite:///{path.as_posix()}'

    # assume SQLAlchemy URL
    if '://' in target:
        return target

    raise SystemExit(f'Target {target!r} is neither an existing path nor a database URL')


def repair(url: str, apply_changes: bool = False, fix_sizes: bool = False) -> dict:
    """Scan all blob attributes and repair corrupted ones. Returns a summary dict."""

    engine = sqlalchemy.create_engine(url)
    summary = {'scanned': 0, 'healthy': 0, 'repaired': 0, 'resized': 0, 'unreadable': []}

    with engine.connect() as conn:
        rows = conn.execute(sqlalchemy.text(
            "SELECT entity_uuid, name FROM attributes WHERE data_type = 'blob'"
        )).fetchall()

        print(f'Found {len(rows)} blob attribute(s)')

        for entity_uuid, name in rows:
            summary['scanned'] += 1

            blob = conn.execute(sqlalchemy.text(
                'SELECT value_blob FROM attributes WHERE entity_uuid = :u AND name = :n'
            ), {'u': entity_uuid, 'n': name}).scalar()

            if blob is None:
                summary['unreadable'].append((entity_uuid, name, 'value_blob is NULL'))
                continue

            try:
                obj = pickle.loads(blob)
            except Exception as e:
                summary['unreadable'].append((entity_uuid, name, f'unpicklable: {e}'))
                continue

            serializer = None
            if isinstance(obj, pd.Series):
                # Corrupted row: pickled insert row instead of the Serializer.
                #  The correct Serializer object is embedded under '__serializer'.
                if '__serializer' in obj.index:
                    serializer = obj['__serializer']
                else:
                    summary['unreadable'].append((entity_uuid, name,
                                                  'pickled Series without __serializer'))
                    continue
            elif type(obj).__name__ == 'Serializer':
                summary['healthy'] += 1
                if not fix_sizes:
                    continue
                serializer = obj
            else:
                summary['unreadable'].append((entity_uuid, name,
                                              f'unexpected object type {type(obj).__name__}'))
                continue

            new_blob = pickle.dumps(serializer)
            new_size = serializer.__sizeof__()

            is_repair = isinstance(obj, pd.Series)
            action = 'REPAIR' if is_repair else 'resize'
            print(f'  {action}: entity={entity_uuid} attribute={name!r} '
                  f'({len(blob)} -> {len(new_blob)} bytes stored, data_size={new_size})')

            if is_repair:
                summary['repaired'] += 1
            else:
                summary['resized'] += 1

            if apply_changes:
                conn.execute(sqlalchemy.text(
                    'UPDATE attributes SET value_blob = :b, data_size = :s, modified = :m '
                    'WHERE entity_uuid = :u AND name = :n'
                ), {'b': new_blob, 's': new_size, 'm': datetime.datetime.now(),
                    'u': entity_uuid, 'n': name})

        if apply_changes:
            conn.commit()

    engine.dispose()
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='entarchy directory, SQLite file or SQLAlchemy URL')
    parser.add_argument('--apply', action='store_true',
                        help='write repairs to the database (default: dry run)')
    parser.add_argument('--fix-sizes', action='store_true',
                        help='also recompute data_size for healthy blob rows')
    args = parser.parse_args(argv)

    url = _resolve_url(args.target)
    print(f'Database: {url.split("@")[-1] if "@" in url else url}')
    if not args.apply:
        print('DRY RUN - no changes will be written (use --apply to write repairs)')

    summary = repair(url, apply_changes=args.apply, fix_sizes=args.fix_sizes)

    print()
    print(f'Scanned:    {summary["scanned"]}')
    print(f'Healthy:    {summary["healthy"]}')
    print(f'Repaired:   {summary["repaired"]}{"" if args.apply else " (dry run)"}')
    if args.fix_sizes:
        print(f'Resized:    {summary["resized"]}{"" if args.apply else " (dry run)"}')
    if summary['unreadable']:
        print(f'UNREADABLE: {len(summary["unreadable"])}')
        for entity_uuid, name, reason in summary['unreadable']:
            print(f'  entity={entity_uuid} attribute={name!r}: {reason}')

    return 0 if not summary['unreadable'] else 1


if __name__ == '__main__':
    sys.exit(main())
