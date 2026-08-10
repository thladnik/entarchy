"""Bring an existing entarchy's link tables up to the current schema.

The `links` table has been present since early on but was never written to - the
feature was sketched and left unimplemented. Its old shape keyed a pair of
entities and pointed at an optional carrier entity:

    links(linker_uuid, linked_uuid, entity_uuid, created, modified)

The current shape makes the carrier entity's uuid the key, names the kind of
link, and allows the same pair to carry several kinds:

    links(link_uuid, link_type, linker_uuid, linked_uuid, created, modified)

and adds `link_types`, the registry of what each kind may connect.

Because nothing ever wrote to the old table, this replaces it rather than
converting it - but it refuses if it finds any rows, so an entarchy that somehow
does hold links is never silently emptied.

Usage:
    python -m entarchy.tools.migrate_links <target> [--apply]

<target> may be an entarchy directory or a full SQLAlchemy URL. Without --apply
the tool only reports what it would change.
"""
from __future__ import annotations

import argparse
import sys

import sqlalchemy

from ..backend.sql import Base, Link, LinkTypeTable
from ._url import _resolve_url


class LinkMigrationError(RuntimeError):
    pass


def inspect(url: str) -> dict:
    """Report what the link tables currently look like."""
    engine = sqlalchemy.create_engine(url)

    try:
        inspector = sqlalchemy.inspect(engine)
        tables = set(inspector.get_table_names())

        state = {
            'has_links': 'links' in tables,
            'has_link_types': 'link_types' in tables,
            'links_columns': [],
            'link_row_count': 0,
        }

        if state['has_links']:
            state['links_columns'] = [column['name']
                                      for column in inspector.get_columns('links')]

            with engine.connect() as connection:
                state['link_row_count'] = connection.execute(
                    sqlalchemy.text('SELECT COUNT(*) FROM links')).scalar() or 0

        state['links_current'] = set(state['links_columns']) >= {'link_uuid', 'link_type'}
        state['needs_migration'] = not (state['links_current'] and state['has_link_types'])

        return state
    finally:
        engine.dispose()


def migrate(url: str, apply_changes: bool = False, verbose: bool = True) -> bool:
    """Replace the link tables. Returns whether anything needed changing."""
    state = inspect(url)

    if verbose:
        print(f'  links table      {"present" if state["has_links"] else "missing"}'
              f'{" (current schema)" if state["links_current"] else ""}')
        print(f'  link_types table {"present" if state["has_link_types"] else "missing"}')
        if state['has_links']:
            print(f'  rows in links    {state["link_row_count"]}')

    if not state['needs_migration']:
        if verbose:
            print('\nLink tables are already current.')
        return False

    # The old table was never written to by any released code path. If it does
    #  hold rows, something unexpected produced them and dropping it would lose
    #  data, so stop rather than guess.
    if state['link_row_count'] > 0 and not state['links_current']:
        raise LinkMigrationError(
            f'The links table holds {state["link_row_count"]} row(s) in the old schema. '
            f'That schema was never written to by entarchy, so this tool will not drop '
            f'it. Inspect the rows and remove them before migrating.')

    if not apply_changes:
        if verbose:
            print('\nWould drop the old links table and create links and link_types '
                  '(dry run).')
        return True

    engine = sqlalchemy.create_engine(url)
    try:
        if state['has_links'] and not state['links_current']:
            print('  DROP TABLE links')
            with engine.begin() as connection:
                connection.execute(sqlalchemy.text('DROP TABLE links'))

        # create_all only adds what is missing, so this creates link_types and,
        #  where it was just dropped, links
        print('  CREATE TABLE link_types, links')
        Base.metadata.create_all(engine, tables=[LinkTypeTable.__table__, Link.__table__])
    finally:
        engine.dispose()

    if verbose:
        print('\nLink tables migrated.')

    return True


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
        print('DRY RUN - no changes will be written (use --apply to migrate)')
    print()

    migrate(url, apply_changes=args.apply)
    return 0


if __name__ == '__main__':
    sys.exit(main())
