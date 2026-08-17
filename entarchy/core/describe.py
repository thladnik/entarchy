"""What an entity or a collection holds, without reading it.

entarchy's reprs deliberately do not read values: one entity can carry hundreds
of megabytes, and it must not be loaded because it was the last expression in a
cell. That rule is what makes a full picture look impossible, and it is not,
because the shape of every value is already stored. Every attribute row carries
its `data_type` and `data_size`, so one indexed query says what is there, of
what type and how big, without decoding anything.

The principle throughout: **show the shape of everything, the value of what is
cheap.** A scalar is its own summary; a blob is reported by type and size and
read only when asked for.
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

# Beyond this many links, listing what they carry costs more than a description
#  should. DISTINCT has to scan them all before it can stop, so a LIMIT would
#  not help - the guard has to be the count. Measured 148 ms at 20 000 links.
LINK_NAME_LIMIT = 5_000

# How much of a long string to show before cutting it
_VALUE_WIDTH = 60

_SECTION_ORDER = ('attributes', 'links', 'media', 'children', 'ancestry')


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
    def ancestry(self) -> pd.DataFrame:
        return self._section('ancestry')

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._sections:
            available = ', '.join(self._sections) or 'none'
            raise KeyError(f'No section "{name}". Present: {available}.')

        return self._sections[name]

    def __contains__(self, name: str) -> bool:
        return name in self._sections

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


def collection_attribute_rows(metadata, total: int) -> pd.DataFrame:
    """The attributes section for a collection.

    One row per name, with the types it is stored as. `entities` is what a
    single entity cannot show: attributes are per entity rather than per type,
    so a name present on a third of a collection is a fact about the data.
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
        rows.append({
            'name': name,
            'type': types[0] if len(types) == 1 else '{' + ', '.join(types) + '}',
            'entities': f'{entry["entities"]} / {total}',
            'bytes': human_bytes(entry['bytes']),
        })

    return pd.DataFrame(rows, columns=['name', 'type', 'entities', 'bytes'])


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
