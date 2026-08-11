"""Collect the query planner statistics SQLite needs.

Without a sqlite_stat1 table SQLite guesses how selective an index is, and for
entarchy's filters it guesses wrong: it drives the query from the entity type,
reading every entity of that type, rather than from the attribute the filter
names. On 27 000 ROIs a selective lookup took 20 ms that way and 0.4 ms with
statistics (see docs/proposals/attribute-storage.md).

Writing keeps them fresh by itself - the SQLite backend runs PRAGMA optimize
when it closes, bounded by PRAGMA analysis_limit - so this is for a database
that has been read a lot and written to rarely, or one whose contents have
changed shape enough that the old statistics mislead. A full ANALYZE reads the
indexes, so it takes about a second per gigabyte.

MySQL is not covered: InnoDB maintains its own statistics.

Usage:
    python -m entarchy.tools.optimize_storage <target> [--apply]

<target> may be an entarchy directory or a full SQLAlchemy URL. Without --apply
the tool only reports what it would do. Safe to repeat.
"""
from __future__ import annotations

import argparse
import time

import sqlalchemy

from ._url import _resolve_url


def inspect(url: str) -> dict:
    """Report whether this database has planner statistics."""
    engine = sqlalchemy.create_engine(url)

    try:
        inspector = sqlalchemy.inspect(engine)

        if 'attributes' not in set(inspector.get_table_names()):
            raise SystemExit(f'{url} has no attributes table - is this an entarchy?')

        state = {'dialect': engine.dialect.name, 'has_statistics': None,
                 'attribute_rows': 0}

        with engine.connect() as connection:
            state['attribute_rows'] = connection.execute(
                sqlalchemy.text('SELECT COUNT(*) FROM attributes')).scalar() or 0

            if engine.dialect.name == 'sqlite':
                # sqlite_stat1 only exists once something has been analysed
                has_table = connection.execute(sqlalchemy.text(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = 'sqlite_stat1'")).scalar()
                state['has_statistics'] = bool(has_table) and bool(connection.execute(
                    sqlalchemy.text('SELECT COUNT(*) FROM sqlite_stat1')).scalar())

        state['needs_work'] = state['has_statistics'] is False

        return state
    finally:
        engine.dispose()


def optimize(url: str, apply_changes: bool = False, verbose: bool = True) -> dict:
    """Run ANALYZE if there are no statistics. Returns what was done."""
    state = inspect(url)
    done = {'analysed': False, 'attribute_rows': state['attribute_rows']}

    if verbose:
        print(f'  dialect                {state["dialect"]}')
        print(f'  attribute rows         {state["attribute_rows"]}')
        if state['has_statistics'] is None:
            print('  planner statistics     maintained by the server')
        else:
            print(f'  planner statistics     '
                  f'{"present" if state["has_statistics"] else "missing"}')

    if not state['needs_work']:
        if verbose:
            print('\nNothing to collect.')
        return done

    if not apply_changes:
        if verbose:
            print('\nWould run ANALYZE to collect planner statistics (dry run).')
        return done

    engine = sqlalchemy.create_engine(url)
    try:
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
        print('\nStatistics collected.')

    return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='entarchy directory or SQLAlchemy URL')
    parser.add_argument('--apply', action='store_true',
                        help='collect the statistics (default: dry run)')
    args = parser.parse_args(argv)

    url = _resolve_url(args.target)
    print(f'Database: {url.split("@")[-1] if "@" in url else url}')
    if not args.apply:
        print('DRY RUN - no changes will be written (use --apply to collect)')
    print()

    optimize(url, apply_changes=args.apply)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
