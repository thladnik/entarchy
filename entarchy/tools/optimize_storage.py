"""Bring an existing entarchy's attribute storage up to the current shape.

Two things, both measured on a 27 000 ROI entarchy (see
docs/proposals/attribute-storage.md):

1.  Drop `ix_unique_name_per_entity_uuid`. The attributes table has a primary
    key on (entity_uuid, name), which every dialect already backs with a unique
    index; this second unique index on the same two columns in the same order
    duplicated it exactly - ANALYZE reported identical statistics for both. It
    cost 44 MB on a 1.12 GB entarchy, 65 bytes per attribute row, and a share of
    every insert. Nothing reads it that the primary key does not serve.

2.  Collect query planner statistics (SQLite only). Without a sqlite_stat1
    table SQLite guesses how selective an index is, and for entarchy's filters
    it guesses wrong: it drives the query from the entity type, reading every
    entity of that type, instead of from the attribute the filter names. A
    selective lookup took 20 ms that way and 0.4 ms with statistics.

    New writes keep the statistics fresh by themselves - the SQLite backend runs
    PRAGMA optimize when it closes - but a database written before that existed
    has none at all, and one full ANALYZE is the way to get them. It reads the
    indexes, so it takes about a second per gigabyte.

MySQL keeps its own statistics, so only step 1 applies there.

Usage:
    python -m entarchy.tools.optimize_storage <target> [--apply]

<target> may be an entarchy directory or a full SQLAlchemy URL. Without --apply
the tool only reports what it would change. Both steps are safe to repeat.
"""
from __future__ import annotations

import argparse
import time

import sqlalchemy

from .repair_blobs import _resolve_url

DUPLICATE_INDEX = 'ix_unique_name_per_entity_uuid'


def inspect(url: str) -> dict:
    """Report what this database still has to gain."""
    engine = sqlalchemy.create_engine(url)

    try:
        inspector = sqlalchemy.inspect(engine)
        is_sqlite = engine.dialect.name == 'sqlite'

        state = {
            'dialect': engine.dialect.name,
            'has_duplicate_index': False,
            'has_statistics': None,
            'attribute_rows': 0,
        }

        if 'attributes' not in set(inspector.get_table_names()):
            raise SystemExit(f'{url} has no attributes table - is this an entarchy?')

        state['has_duplicate_index'] = any(
            index['name'] == DUPLICATE_INDEX for index in inspector.get_indexes('attributes'))

        with engine.connect() as connection:
            state['attribute_rows'] = connection.execute(
                sqlalchemy.text('SELECT COUNT(*) FROM attributes')).scalar() or 0

            if is_sqlite:
                # sqlite_stat1 only exists once something has been analysed
                has_table = connection.execute(sqlalchemy.text(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = 'sqlite_stat1'")).scalar()
                state['has_statistics'] = bool(has_table) and bool(connection.execute(
                    sqlalchemy.text('SELECT COUNT(*) FROM sqlite_stat1')).scalar())

        state['needs_work'] = (state['has_duplicate_index']
                               or state['has_statistics'] is False)

        return state
    finally:
        engine.dispose()


def optimize(url: str, apply_changes: bool = False, verbose: bool = True) -> dict:
    """Drop the duplicate index and collect statistics. Returns what was done."""
    state = inspect(url)
    done = {'dropped_index': False, 'analysed': False,
            'attribute_rows': state['attribute_rows']}

    if verbose:
        print(f'  dialect                {state["dialect"]}')
        print(f'  attribute rows         {state["attribute_rows"]}')
        print(f'  duplicate index        '
              f'{"present" if state["has_duplicate_index"] else "already gone"}')
        if state['has_statistics'] is None:
            print('  planner statistics     maintained by the server')
        else:
            print(f'  planner statistics     '
                  f'{"present" if state["has_statistics"] else "missing"}')

    if not state['needs_work']:
        if verbose:
            print('\nAttribute storage is already current.')
        return done

    if not apply_changes:
        if verbose:
            print()
            if state['has_duplicate_index']:
                print(f'Would drop {DUPLICATE_INDEX} (dry run).')
            if state['has_statistics'] is False:
                print('Would run ANALYZE to collect planner statistics (dry run).')
        return done

    engine = sqlalchemy.create_engine(url)
    try:
        if state['has_duplicate_index']:
            if verbose:
                print(f'  DROP INDEX {DUPLICATE_INDEX}')
            with engine.begin() as connection:
                # MySQL has no DROP INDEX ... on its own; both dialects accept
                #  the table-qualified form through ALTER, but SQLite does not,
                #  so each gets its own spelling
                if engine.dialect.name == 'sqlite':
                    connection.execute(sqlalchemy.text(f'DROP INDEX {DUPLICATE_INDEX}'))
                else:
                    connection.execute(sqlalchemy.text(
                        f'ALTER TABLE attributes DROP INDEX {DUPLICATE_INDEX}'))
            done['dropped_index'] = True

        if state['has_statistics'] is False:
            if verbose:
                print('  ANALYZE')
            start = time.perf_counter()
            with engine.begin() as connection:
                connection.execute(sqlalchemy.text('ANALYZE'))
            done['analysed'] = True
            done['analyse_seconds'] = time.perf_counter() - start
            if verbose:
                print(f'    collected in {done["analyse_seconds"]:.1f} s')
    finally:
        engine.dispose()

    if verbose:
        print('\nAttribute storage optimised.')
        if done['dropped_index'] and state['dialect'] == 'sqlite':
            print('The space the index held is on the free list; run VACUUM to '
                  'return it to the filesystem.')

    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='entarchy directory or SQLAlchemy URL')
    parser.add_argument('--apply', action='store_true',
                        help='make the changes (default: dry run)')
    args = parser.parse_args(argv)

    url = _resolve_url(args.target)
    print(f'Database: {url.split("@")[-1] if "@" in url else url}')
    if not args.apply:
        print('DRY RUN - no changes will be written (use --apply to optimise)')
    print()

    optimize(url, apply_changes=args.apply)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
