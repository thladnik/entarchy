"""Give the timestamp columns of an existing MySQL entarchy fractional seconds.

MySQL's DATETIME keeps whole seconds and rounds on insert, so a row written at
12:30:05.7 is stored as 12:30:06. Collections select entities with
"created <= collection init time", so entities added moments earlier fall out of
every query for up to a second, depending on where in the second the write landed.

Entarchies created before this was fixed have DATETIME columns without fractional
seconds and keep that behaviour until the columns are altered. This tool alters
them in place. No data is lost: existing values simply gain a zero fraction.

SQLite stores timestamps at full precision already, so this is a no-op there.

Usage:
    python -m entarchy.tools.migrate_datetime_precision <target> [--apply]

<target> may be an entarchy directory, or a full SQLAlchemy URL. Without --apply
the tool only reports what it would change.
"""
from __future__ import annotations

import argparse
import sys

import sqlalchemy

from .repair_blobs import _resolve_url

# Columns that carry a timestamp, and therefore need fractional seconds
TIMESTAMP_COLUMNS = [
    ('entities', 'created'),
    ('entities', 'modified'),
    ('links', 'created'),
    ('links', 'modified'),
    ('attributes', 'created'),
    ('attributes', 'modified'),
    ('attributes', 'value_datetime'),
]

TARGET_PRECISION = 6


def inspect(url: str) -> list[dict]:
    """Report the current precision of every timestamp column."""
    engine = sqlalchemy.create_engine(url)
    findings = []

    try:
        with engine.connect() as connection:
            if connection.dialect.name != 'mysql':
                print(f'Backend is {connection.dialect.name}, which stores timestamps at full '
                      f'precision already. Nothing to do.')
                return []

            schema = connection.engine.url.database

            for table, column in TIMESTAMP_COLUMNS:
                row = connection.execute(sqlalchemy.text(
                    'SELECT DATETIME_PRECISION, IS_NULLABLE, COLUMN_TYPE '
                    'FROM information_schema.COLUMNS '
                    'WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table '
                    'AND COLUMN_NAME = :column'
                ), {'schema': schema, 'table': table, 'column': column}).fetchone()

                if row is None:
                    continue

                precision, nullable, column_type = row
                findings.append({
                    'table': table,
                    'column': column,
                    'precision': precision or 0,
                    'nullable': nullable == 'YES',
                    'type': column_type,
                    'needs_migration': (precision or 0) < TARGET_PRECISION,
                })
    finally:
        engine.dispose()

    return findings


def migrate(url: str, apply_changes: bool = False) -> int:
    """Alter the timestamp columns. Returns the number of columns changed."""
    findings = inspect(url)
    if not findings:
        return 0

    outstanding = [f for f in findings if f['needs_migration']]

    for finding in findings:
        state = 'needs migration' if finding['needs_migration'] else 'ok'
        label = f'{finding["table"]}.{finding["column"]}'
        print(f'  {label:<28} {finding["type"]:<12} precision {finding["precision"]} - {state}')

    if not outstanding:
        print('\nAll timestamp columns already keep fractional seconds.')
        return 0

    if not apply_changes:
        print(f'\n{len(outstanding)} column(s) would be altered (dry run).')
        return len(outstanding)

    engine = sqlalchemy.create_engine(url)
    try:
        with engine.connect() as connection:
            for finding in outstanding:
                null_clause = 'NULL' if finding['nullable'] else 'NOT NULL'
                statement = (f'ALTER TABLE {finding["table"]} '
                             f'MODIFY {finding["column"]} DATETIME({TARGET_PRECISION}) {null_clause}')
                print(f'  {statement}')
                connection.execute(sqlalchemy.text(statement))
            connection.commit()
    finally:
        engine.dispose()

    print(f'\nAltered {len(outstanding)} column(s).')
    return len(outstanding)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='entarchy directory or SQLAlchemy URL')
    parser.add_argument('--apply', action='store_true',
                        help='alter the columns (default: dry run)')
    args = parser.parse_args(argv)

    url = _resolve_url(args.target)
    print(f'Database: {url.split("@")[-1] if "@" in url else url}')
    if not args.apply:
        print('DRY RUN - no changes will be written (use --apply to migrate)')
    print()

    migrate(url, apply_changes=args.apply)
    return 0


if __name__ == '__main__':
    sys.exit(main())
