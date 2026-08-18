"""What an entarchy, an entity or a collection holds, without reading it.

entarchy's reprs deliberately do not read values: one entity can carry hundreds
of megabytes, and it must not be loaded because it was the last expression in a
cell. That rule is what makes a full picture look impossible, and it is not,
because the shape of every value is already stored. Every attribute row carries
its `data_type` and `data_size`, so one indexed query says what is there, of
what type and how big, without decoding anything.

The principle throughout: **show the shape of everything, the value of what is
cheap.** A scalar is its own summary; a blob is reported by type and size and
read only when asked for.

Two things here do have to look at the data rather than at its shape, and both
are for that reason opt-in or capped. Totalling bytes for
`Entarchy.describe()` scans the attributes table, since there is no adding up
what has not been looked at. `Collection.describe(distribution=True)` asks the
database for a minimum, a maximum and a distinct count, which is a query per
stored type - and which takes the server's collation for text, so the same
strings can count differently on SQLite and on MySQL.
"""
from __future__ import annotations

import html
from typing import Any

import pandas as pd

# Values are inlined when they are cheap; anything else is described instead
SCALAR_TYPES = ('str', 'int', 'float', 'bool', 'date', 'datetime')

# Stored as attribute rows like any other, but they are what an entity *is*
#  rather than anything it holds: both are in the headline already, and on a
#  link they are not even consistently present - a link made one at a time gets
#  them and one written in bulk does not.
BOOKKEEPING_NAMES = ('id', 'uuid')

# Every link is carried by an entity of this type, parented to its linker. That
#  keeps the entity tree valid and gives archives and map_async the grouping
#  they use everywhere else, at the price of links turning up wherever entities
#  are counted. They have their own section; this is the name that keeps them
#  out of the others.
LINK_ENTITY_TYPE = 'LinkEntity'

# Beyond this many links, listing what they carry costs more than a description
#  should. DISTINCT has to scan them all before it can stop, so a LIMIT would
#  not help - the guard has to be the count. Measured 148 ms at 20 000 links.
LINK_NAME_LIMIT = 5_000

# How many rows of the storage section to show before saying how many were cut.
#  It is a ranking rather than a partition, so a cut loses the tail rather than
#  making the numbers stop adding up.
STORAGE_ROWS = 15

# How much of a long string to show before cutting it
_VALUE_WIDTH = 60

_SECTION_ORDER = ('entities', 'attributes', 'links', 'media',
                  'children', 'storage', 'ancestry')


def human_bytes(count: int) -> str:
    """A byte count as something a reader takes in at a glance."""
    if count is None:
        return ''

    size = float(count)
    for unit in ('B', 'kB', 'MB', 'GB'):
        if abs(size) < 1024 or unit == 'GB':
            return f'{size:.0f} {unit}' if unit == 'B' else f'{size:.1f} {unit}'
        size /= 1024


def short_value(value: Any) -> str:
    """A value as one line, cut if it runs long.

    Cut with three dots rather than an ellipsis character: a description is
    printed to whatever console is there, and a Windows one on a legacy code
    page raises on the ellipsis rather than showing it.
    """
    text = str(value)
    if len(text) > _VALUE_WIDTH:
        return f'{text[:_VALUE_WIDTH - 3]}...'

    return text


class Description:
    """What one entity or collection holds, in sections.

    Each section is a DataFrame and can be used as one; the whole renders
    together, as a table in a terminal and as HTML in a notebook.

        description = roi.describe()
        description.attributes          # a DataFrame
        description                     # all of it

    Sections that would be empty are left out, so an entarchy that uses no links
    reads as though links did not exist.
    """

    def __init__(self, subject: str, headline: dict[str, Any],
                 sections: dict[str, pd.DataFrame], notes: list[str] = None):
        self._subject = subject
        self._headline = headline
        self._sections = {name: frame for name, frame in sections.items()
                          if frame is not None and len(frame) > 0}
        self._notes = list(notes or [])

    @property
    def subject(self) -> str:
        return self._subject

    @property
    def headline(self) -> dict[str, Any]:
        """Type, id, uuid, path - whatever identifies what is being described."""
        return dict(self._headline)

    @property
    def sections(self) -> dict[str, pd.DataFrame]:
        return dict(self._sections)

    @property
    def notes(self) -> list[str]:
        """What was capped, skipped or is worth knowing about this description."""
        return list(self._notes)

    def _section(self, name: str) -> pd.DataFrame:
        return self._sections.get(name, pd.DataFrame())

    @property
    def entities(self) -> pd.DataFrame:
        return self._section('entities')

    @property
    def attributes(self) -> pd.DataFrame:
        return self._section('attributes')

    @property
    def links(self) -> pd.DataFrame:
        return self._section('links')

    @property
    def media(self) -> pd.DataFrame:
        return self._section('media')

    @property
    def children(self) -> pd.DataFrame:
        return self._section('children')

    @property
    def storage(self) -> pd.DataFrame:
        return self._section('storage')

    @property
    def ancestry(self) -> pd.DataFrame:
        return self._section('ancestry')

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._sections:
            available = ', '.join(self._sections) or 'none'
            raise KeyError(f'No section "{name}". Present: {available}.')

        return self._sections[name]

    def __contains__(self, name: str) -> bool:
        return name in self._sections

    def _ipython_key_completions_(self) -> list[str]:
        """The section names, for tab completion inside the brackets.

        Which sections a description has depends on what the thing being
        described turned out to hold, so this is the only way to find out
        without printing it first.
        """
        return list(self._sections)

    def _ordered_sections(self):
        known = [n for n in _SECTION_ORDER if n in self._sections]
        rest = [n for n in self._sections if n not in _SECTION_ORDER]
        return known + rest

    def __repr__(self) -> str:
        """A text rendering, because the terminal is not a notebook."""
        try:
            lines = [self._subject]
            lines += [f'  {key}: {value}' for key, value in self._headline.items()]

            for name in self._ordered_sections():
                frame = self._sections[name]
                lines.append('')
                lines.append(f'{name} ({len(frame)})')
                body = frame.to_string(index=False, max_rows=40)
                lines += [f'  {line}' for line in body.splitlines()]

            if len(self._sections) == 0:
                lines.append('')
                lines.append('nothing to show')

            for note in self._notes:
                lines.append(f'! {note}')

            return '\n'.join(lines)
        except Exception as exc:
            # A description is reached for when something is already confusing.
            #  It has to degrade rather than fail.
            return f'{self._subject} (description could not be rendered: {exc})'

    def _repr_html_(self) -> str:
        try:
            head = ''.join(
                f'<tr><td style="text-align:right;color:#888;padding-right:8px">'
                f'{html.escape(str(key))}</td>'
                f'<td style="font-family:monospace">{html.escape(str(value))}</td></tr>'
                for key, value in self._headline.items())

            parts = [f'<div><b>{html.escape(self._subject)}</b>'
                     f'<table style="border:none">{head}</table>']

            for name in self._ordered_sections():
                frame = self._sections[name]
                parts.append(
                    f'<div style="margin-top:8px;color:#888;font-size:90%">'
                    f'{html.escape(name)} &middot; {len(frame)}</div>')
                parts.append(frame.to_html(index=False, escape=True,
                                           max_rows=40, border=0))

            if len(self._sections) == 0:
                parts.append('<div style="color:#888;font-size:90%">'
                             'nothing to show</div>')

            for note in self._notes:
                parts.append(f'<div style="color:#888;font-size:90%">'
                             f'{html.escape(note)}</div>')

            parts.append('</div>')
            return ''.join(parts)
        except Exception:
            from .entity import _fallback_html

            return _fallback_html(self)


def attribute_rows(metadata, values: dict = None,
                   media_names: list[str] = None) -> pd.DataFrame:
    """The attributes section for one entity.

    `id` and `uuid` are left out: they are in the headline, and repeating them
    here as though they were data would put two rows of nothing on every
    description there is.

    A media file is stored as a blob like anything else, but saying `blob` of
    it would send a reader looking for an array. It is named `media` here and
    detailed in its own section.

    Args:
        metadata: (name, data_type, data_size) as the backend gives it.
        values: already-read values for the names worth inlining, or None.
        media_names: which names are media files.
    """
    values = values or {}
    media = set(media_names or ())
    rows = []

    for name, data_type, data_size in metadata:
        if name in BOOKKEEPING_NAMES:
            continue

        rows.append({
            'name': name,
            'type': 'media' if name in media else (data_type or ''),
            'bytes': human_bytes(data_size),
            'value': (short_value(values[name]) if name in values else ''),
        })

    return pd.DataFrame(rows, columns=['name', 'type', 'bytes', 'value'])


def _bounded(value: Any) -> Any:
    """A range endpoint, kept as the value it is.

    Numbers, dates and booleans go through untouched, so a reader can compare
    against them rather than against a rendering of them. Only text is cut, and
    only because `value_str` is a TEXT column and one long value would otherwise
    take the table with it.
    """
    return short_value(value) if isinstance(value, str) else value


def _range_of(name: str, types: list[str], distribution: dict) -> dict[str, Any]:
    """The lowest, the highest and how many different values a name holds.

    Across the types the name is stored as, because a name written as `int` on
    some entities and `float` on others has a range in each and one range is
    what a reader wants. Infinity is folded back in - it is stored as a flag
    with a null value column, so the query cannot see it, and a maximum that
    left out an infinity would simply be wrong.

    NaN is counted as one more distinct value and kept out of the range, having
    no place in an ordering. `Collection.describe()` says so in a note, since a
    range with a silent NaN behind it invites the wrong conclusion.

    Text ranges are the database's, not Python's: MIN, MAX and COUNT(DISTINCT)
    take the server's collation, and SQLite compares bytes where MySQL 8
    defaults to case- and accent-insensitive. The same strings therefore give
    the same numbers within one entarchy and can differ between two. Numbers,
    booleans and times agree everywhere.

    Which is why one type's two ends are kept as the pair the database gave and
    never re-derived here. Re-minimising them in Python would answer in byte
    order instead: MySQL calls 'abc' the least of ['abc', 'ABC', 'Zeta'] and
    'Zeta' the greatest, and Python, handed just those two, would call 'Zeta'
    the least and report a range the wrong way round. Only several types get
    compared against each other, and the only combinations Python will compare
    at all are the numeric ones, where it and the server agree.
    """
    lows, highs, distinct = [], [], 0

    for data_type in types:
        entry = distribution.get((name, data_type))
        if entry is None:
            continue

        # NaN and each infinity are values the query could not count, because
        #  each is stored as a flag with a null value column
        distinct += (entry['distinct'] + bool(entry['nan'])
                     + bool(entry['plus_inf']) + bool(entry['minus_inf']))

        # And for the same reason an infinity is not in what the query
        #  returned, and has to be put back at its end
        low = float('-inf') if entry['minus_inf'] else entry['min']
        high = float('inf') if entry['plus_inf'] else entry['max']

        if low is None and high is None:
            # Nothing here that an ordering can hold: either no values at all,
            #  or nothing but NaN, which is counted above and stops there
            continue

        # An attribute whose only value is one infinity has it at both ends
        low = high if low is None else low
        high = low if high is None else high

        lows.append(low)
        highs.append(high)

    if len(lows) == 0:
        return {'min': '', 'max': '', 'distinct': distinct or ''}

    try:
        low, high = _bounded(min(lows)), _bounded(max(highs))
    except TypeError:
        # A name stored as both a string and a number has no single range, and
        #  Python says so by refusing to compare them. The type column already
        #  shows the reader why this is blank.
        low, high = '', ''

    return {'min': low, 'max': high, 'distinct': distinct}


def collection_attribute_rows(metadata, total: int,
                              distribution: dict = None) -> pd.DataFrame:
    """The attributes section for a collection.

    One row per name, with the types it is stored as. `entities` is what a
    single entity cannot show: attributes are per entity rather than per type,
    so a name present on a third of a collection is a fact about the data.

    Args:
        metadata: (name, data_type, entity_count, total_size) rows.
        total: how many entities the collection has, for the coverage column.
        distribution: (name, data_type) -> range, if the caller asked for one.
            Adds min, max and distinct; blobs stay blank, having none of the
            three without being decoded.
    """
    merged: dict[str, dict] = {}

    for name, data_type, count, total_size in metadata:
        if name in BOOKKEEPING_NAMES:
            continue
        entry = merged.setdefault(name, {'types': set(), 'entities': 0, 'bytes': 0})
        entry['types'].add(data_type or '')
        entry['entities'] += count
        entry['bytes'] += total_size

    rows = []
    for name, entry in sorted(merged.items()):
        types = sorted(entry['types'])
        row = {
            'name': name,
            'type': types[0] if len(types) == 1 else '{' + ', '.join(types) + '}',
            'entities': f'{entry["entities"]} / {total}',
            'bytes': human_bytes(entry['bytes']),
        }
        if distribution is not None:
            row.update(_range_of(name, types, distribution))

        rows.append(row)

    columns = ['name', 'type', 'entities', 'bytes']
    if distribution is not None:
        columns += ['min', 'max', 'distinct']

    return pd.DataFrame(rows, columns=columns)


def link_rows(counts: dict[str, int],
              carried: dict[str, list[str]] = None) -> pd.DataFrame:
    """The links section: kind, how many, and what each kind carries.

    `id` and `uuid` are dropped from what a kind carries: they are entity
    bookkeeping stored as attribute rows rather than anything the link is
    about, and they are only present at all for links created one at a time.
    """
    carried = carried or {}
    rows = []

    for link_type, count in sorted(counts.items()):
        names = [n for n in carried.get(link_type, [])
                 if n not in BOOKKEEPING_NAMES]
        rows.append({
            'kind': link_type,
            'links': count,
            'carries': ', '.join(names) if names else '',
        })

    return pd.DataFrame(rows, columns=['kind', 'links', 'carries'])


def media_rows(entity, names: list[str], verify: bool = False) -> pd.DataFrame:
    """The media section.

    Never verifies unless asked: `MediaFile.verify()` re-reads the whole file to
    re-hash it, which is 114 MB for one behaviour video. `exists()` is a stat.
    """
    rows = []

    for name in sorted(names):
        try:
            media = entity[name]
            present = media.exists()
            row = {
                'name': name,
                'media type': media.media_type or '',
                'bytes': human_bytes(media.bytes),
                'present': present,
            }
            if verify:
                row['verified'] = bool(present and media.verify())
        except Exception as exc:
            row = {'name': name, 'media type': '', 'bytes': '',
                   'present': f'unreadable: {type(exc).__name__}'}
            if verify:
                row['verified'] = False

        rows.append(row)

    columns = ['name', 'media type', 'bytes', 'present']
    if verify:
        columns.append('verified')

    return pd.DataFrame(rows, columns=columns)


def count_rows(counts: dict[str, int], key: str, value: str) -> pd.DataFrame:
    """A plain two column section, for children and the like."""
    return pd.DataFrame([{key: name, value: count}
                         for name, count in sorted(counts.items())],
                        columns=[key, value])


def hierarchy_order(hierarchy: dict) -> list[tuple[str, str]]:
    """Every declared entity type with its parent, outermost first.

    Depth first through the declared hierarchy, so a census of the types reads
    the way the schema does. Sorted alphabetically it would put Roi above
    Recording and tell a stranger nothing about which contains which.
    """
    order = []

    def walk(level: dict, parent: str):
        for name, children in level.items():
            order.append((name, parent))
            walk(children, name)

    walk(hierarchy or {}, '')

    return order


def entity_type_rows(counts: dict[str, int], sizes: dict[str, int],
                     hierarchy: dict = None) -> pd.DataFrame:
    """The census: how many entities of each type there are, and what they cost.

    Link carriers are left out. They are entities, and every link in the
    entarchy is one, so counting them here would put a number nobody wrote
    under a type name nobody declared, next to the ones they did. The links
    section counts them, by kind.

    A declared type with no entities is left out too: this is what is here, and
    the hierarchy is what says what could be.
    """
    declared = hierarchy_order(hierarchy)
    parents = dict(declared)
    ordered = [name for name, _ in declared]

    # A type in the data but not in the hierarchy is a real disagreement between
    #  the two, so it goes last rather than going missing
    rest = sorted(name for name in counts if name not in parents)

    rows = [{'type': name,
             'parent': parents.get(name, ''),
             'entities': counts[name],
             'bytes': human_bytes(sizes.get(name, 0))}
            for name in ordered + rest
            if name in counts and name != LINK_ENTITY_TYPE]

    return pd.DataFrame(rows, columns=['type', 'parent', 'entities', 'bytes'])


def link_type_rows(specs, totals: dict[str, dict[str, int]]) -> pd.DataFrame:
    """The link kinds: what each joins, how many there are and what they cost.

    `between` carries the direction. `->` for a directed kind, `--` for a
    symmetric one, where both ends are the same type and which is the linker
    means nothing.

    A kind that is registered but unused still gets a row, at zero. Kinds are
    invented at runtime and the registry is the only schema there is, so a
    declared kind with no links is worth meeting rather than worth hiding.
    """
    rows = []
    seen = set()

    for spec in sorted(specs, key=lambda s: s.name):
        seen.add(spec.name)
        total = totals.get(spec.name, {})
        arrow = '--' if spec.symmetric else '->'
        rows.append({
            'kind': spec.name,
            'between': f'{spec.linker.describe()} {arrow} {spec.linked.describe()}',
            'cardinality': spec.cardinality,
            'links': total.get('links', 0),
            'bytes': human_bytes(total.get('bytes', 0)),
        })

    # Links of a kind the registry has never heard of should not exist - and a
    #  description is reached for when something is already wrong
    for name in sorted(set(totals) - seen):
        rows.append({'kind': name, 'between': 'unregistered', 'cardinality': '',
                     'links': totals[name].get('links', 0),
                     'bytes': human_bytes(totals[name].get('bytes', 0))})

    return pd.DataFrame(rows, columns=['kind', 'between', 'cardinality',
                                       'links', 'bytes'])


def storage_rows(storage, limit: int = STORAGE_ROWS) -> tuple[pd.DataFrame, int]:
    """Where the bytes are: the largest attributes, largest first.

    A ranking rather than a partition, so cutting it loses the tail rather than
    making the numbers stop adding up. `id` and `uuid` are left in, unlike
    everywhere else a description mentions them: they are not data, but they
    are bytes, and this is the section about bytes.

    Returns the rows and how many were left off, so the caller can say so.
    """
    ranked = sorted(storage, key=lambda row: row[4], reverse=True)
    shown = ranked if limit is None else ranked[:limit]

    rows = [{'entity type': type_name, 'name': name, 'type': data_type or '',
             'entities': count, 'bytes': human_bytes(total)}
            for type_name, name, data_type, count, total in shown]

    return (pd.DataFrame(rows, columns=['entity type', 'name', 'type',
                                        'entities', 'bytes']),
            len(ranked) - len(shown))
