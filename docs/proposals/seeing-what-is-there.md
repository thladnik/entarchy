# Proposal: seeing what an entity or collection holds

Status: **implemented**, steps 1–5, plus two of the three questions this left
open. `Entity.describe()`, `Collection.describe()`, `Entarchy.describe()`, the
`Description` object and both reprs are in place, and `preview()` no longer
reads blobs. Step 6's `verify=True` is in; `links=True` for link *contents* is
not, because the links section turned out to say enough without them — see
"What changed in the building".

On the entarchy this was measured against:

| | |
|---|---|
| `Recording.describe()` | 0.38 s — against `preview(3)` reading 653 MB |
| `Phase.describe()` | 0.01 s |
| `describe()` over 42,521 ROIs | 4.56 s |

The first useful thing it said, unprompted: `ants/x` is on **33,468 of 42,521**
ROIs. The 9,053 without it are exactly the tailtracking ones, which have no
registration output — a fact about the data that took a column to notice.

Measured against: entarchy at `089c45a`, on the vxpy entarchy at
`E:/data/entarchy_vxpy` — 8 animals, 38 recordings, 42,521 ROIs, 1.6 million
attribute rows. SQLite 3.40.1, Windows 11 / Python 3.10.

## Summary

There is no way to ask an entity or a collection what it holds. The pieces exist
and none of them is a whole answer:

| | attributes | links | media | children |
|---|---|---|---|---|
| `Entity._repr_html_()` | names only, capped at 40 | counts per kind | not distinguished | — |
| `Entity.keys()` | names | — | — | — |
| `Entity.to_dict()` | **every value**, however large | — | as values | — |
| `Collection._repr_html_()` | — | — | — | — |
| `Collection.preview(n)` | see below | — | — | — |
| `Collection.columns` | names | — | — | — |

So: to find out that a Recording carries 43 blobs totalling several hundred
megabytes, or that a Phase has a `phase_frames` link to each imaging source and
what is in it, you have to know to ask — and know which of six accessors to ask
with.

This proposes `describe()` on both, returning something that renders as a whole
picture and can also be indexed into.

## The thing to fix first

`Collection.preview()` says this:

> Attributes to include. Defaults to the scalar attributes of the collection, so
> large blobs are not loaded.

It does the opposite. `attribute_names` defaults to `subset.columns`, which is
every attribute name the collection has, blobs included:

| `preview(3)` on | columns | time | ndarray pulled |
|---|---|---|---|
| `Roi` | 28 | 12.9 s | 0.1 MB |
| `Imaging` | 7 | 2.7 s | 0.1 MB |
| `Layer` | 138 | 8.5 s | **196 MB** |
| `Recording` | 82 | 10.0 s | **653 MB** |

Three rows of `Recording` reads two thirds of a gigabyte, because
`camera/__time`, `io/ai_y_mirror_in` and forty other blobs are columns like any
other. The one thing a preview must not do is exactly what it does.

**This is worth fixing whether or not the rest is built**, and the fix is small:
ask `get_attribute_data_types` which names are blobs and leave them out by
default, naming them in the result so their absence is visible rather than
silent. It is the same shape of defect as the `dataframe_of` one — a docstring
describing an intention the code never had.

## The principle

entarchy already decided that a repr must not read values: `Entity._repr_html_`
says so, and lists names without touching them. Any "show me everything" has to
keep that promise, which sounds like it rules the whole thing out.

It does not, because **the shape of every value is already stored**. Every
attribute row carries `data_type` and `data_size`, and they are populated — 193
of 1,625,093 rows have a zero size, and those are the genuinely empty ones. So
one indexed query answers "what is here, of what type, how big" for an entity
without decoding a single blob:

```
io/__time                              blob        1,761,503
io/ai_y_mirror_in_time                 blob        1,761,503
io/ai_y_mirror_in                      blob          863,575
camera/__time                          blob          717,120
```

The principle, then: **show the shape of everything, the value of what is
cheap.** A scalar is its own summary and costs nothing to include. A blob is
reported by type and size, and read only when asked for.

Note `data_size` is bytes *as stored* — the encoded, compressed container —
which is the honest number for "what does this cost me", and is not the size in
memory. `io/__time` above is 1.76 MB stored against 4.5 MB of float64.

## The shape of it

`describe()` returns a `Description`: sections that are DataFrames, and a repr
that renders all of them.

```python
description = roi.describe()

description.attributes   # name, type, bytes, value        (values for scalars)
description.links        # kind, count, direction, attribute names it carries
description.media        # name, media type, bytes, present
description.children     # type, count
description.ancestry     # type, id  — up to the root

description              # renders whole, in a notebook
print(description)       # renders whole, in a terminal
```

### What the links section says without reading links

There is a middle ground between "327 links of kind `phase_frames`" and reading
327 link entities, and it is the more useful half: **which attribute names those
links carry.** A kind's names are its shape — `phase_frames` carries
`start_index` and `end_index`, a correlation carries `r` and `p_value` — and
that is what a reader wants when they meet a kind they did not create.

It costs a `DISTINCT` over the links touching the entity, joined to the
attributes table. Both `linker_uuid` and `linked_uuid` are indexed and the
attributes primary key is `(entity_uuid, name)`, so it rides indexes throughout,
and it scales with *this entity's* links rather than with the entarchy's:

| entity | links | `count_links_by_type` | + attribute names |
|---|---|---|---|
| a Phase | 1 | 1.6 ms | 1.7 ms |
| an Imaging | 327 | 1.4 ms | 7.8 ms |
| a synthetic hub | 20,000 | 23.0 ms | 147.7 ms |

So the default can be: kinds, counts, and the attribute names each kind carries.
147 ms in the pathological case is acceptable for something a reader asked for
by name; if it needs a guard, the guard is a link-count threshold above which
the names are skipped and said to be skipped — **not** a `LIMIT` on the query,
which does not help. `DISTINCT` has to scan everything before it can stop, and
`LIMIT 50` measured 148.8 ms against 147.7 ms for the full answer.

The entarchy-wide version of the same question — what does kind *k* carry
anywhere — is a different matter: **726 ms** for `phase_frames`, because it
scans all 17,079 of them. That belongs behind an explicit ask, or on the link
type registry, or nowhere.

**A wrinkle this turned up.** The names come back as
`['end_index', 'id', 'start_index', 'uuid']`. `id` and `uuid` are entity
bookkeeping stored as attribute rows, not the link's payload, so a description
has to filter them. Worse, they are only there for links made one at a time:
`ent.link()` builds a full entity and gets them, while `link_from_frame()`
writes through the core and does not, so the same kind of link has different
attribute names depending on how it was created. Cosmetic for a description,
which filters both away — but it is an inconsistency between the two write
paths, and worth a look on its own.

For a collection the same sections, asked of the set rather than of one entity:

```python
rois.describe().attributes
# name, type, bytes (total), entities (how many have it), example
```

`entities` is the useful column a single entity cannot show: attributes are per
entity rather than per type, so `ants/x` on 34,000 of 42,521 ROIs is a fact
about the data worth seeing without going looking for it.

### Why not `Entity.preview()`

The asymmetry is real — `Collection` has `preview()` and `Entity` has nothing —
but it is the right asymmetry. A preview is a few of many; an entity is one, and
has nothing to take a few of. What is missing from `Entity` is not a preview but
a description, and giving both types `describe()` makes the pair symmetric in
the way that matters. `Collection.preview()` keeps its meaning: the first *n*
rows, in the collection's order.

### Naming

`describe()` collides with the pandas meaning, which is summary *statistics*,
and this audience lives in pandas. `info()` is closer to the intent — pandas'
`DataFrame.info()` reports structure — but it prints and returns None, which is
a bad model for something that should be indexable.

Recommendation: `describe()`, because it returns a value rather than printing,
and because the sections are named plainly enough that nobody will mistake
`description.attributes` for statistics. Worth deciding explicitly rather than
drifting into.

## Pitfalls

**1. Cost is in the number of columns, not their size.** Leaving blobs out of
`preview()` fixes the megabytes but not the 138 `MAX(CASE …)` branches a wide
Layer builds. A `Recording` preview restricted to scalars still pivots 39
columns. Worth measuring after the fix rather than assuming it is solved, and
worth a default cap on how many columns a preview reads, named in the output.

**2. Links can be unbounded.** A ROI in a correlation analysis can have
thousands. `describe()` must cap what it *lists* and say that it capped —
`link_counts()` is the cheap exact total, so the counts stay right while the
listing does not pretend to be complete. Silently showing the first ten of four
thousand would be worse than showing none.

**3. Reading a link's attributes means reading link entities.** The counts come
from an indexed `GROUP BY`; the *contents* do not. Showing the values of every
link is real work per link, and that has to be opt-in (`roi.describe(links=True)`)
rather than the default for an entity that might have thousands.

**4. Media must not be verified.** `MediaFile.verify()` re-hashes the whole
file — 114 MB for one behaviour video. `exists()` is a stat call and is the
right check for a description; `verify()` belongs behind an explicit ask.

**5. An attribute's type is not always one type.** This entarchy has real cases:
`s2p/classifier_path` is `str` on some ROIs and `int` on others,
`s2p/do_registration` is `bool` and `int`. Today that surfaces as a
`RuntimeWarning` from inside a DataFrame read. A description is exactly where it
should be shown as a fact — a `types` column holding `{str, int}` — instead of
warned about somewhere else.

**6. Counting children costs a query per child type.** `get_child_entity_types()`
is a class-level declaration and free; the counts are not. For a Recording with
two child types that is two queries; for a description of a collection of 38
Recordings it must not become 76.

**7. `to_dict()` is the existing footgun and stays one.** It reads every value of
every attribute — on a Recording, the same 650 MB. Out of scope to change, but
if `describe()` exists then `to_dict()`'s docstring should point at it, since
"give me everything" is what people reach for `to_dict()` to mean.

**8. A description must never raise.** The reprs already take this seriously —
both wrap in `try/except` and fall back, because a repr that throws makes a
notebook session unusable. A description is reached for when something is
already confusing; it has to degrade to partial output rather than fail.

**9. The terminal is not a notebook.** `_repr_html_` only fires under Jupyter.
The vxpy work happens in both, so `__repr__` has to render a usable text table
rather than `<Description object at 0x…>`.

## What changed in the building

**Three things had to be filtered that the proposal did not anticipate**, all of
them the same shape: storage detail showing through as though it were data.

- `id` and `uuid` are stored as attribute rows, so every attributes section
  opened with two rows repeating the headline. Filtered, and named as
  `BOOKKEEPING_NAMES` so the links section filters the same two for the same
  reason.
- A link's carrier entity is parented to its linker, which keeps the entity tree
  valid — and made a ROI with five links report five `LinkEntity` children.
  Links have their own section; they are not children.
- Every entity's ancestry began with the entarchy root, which every entity has
  and which therefore says nothing. Dropped, so a top-level entity has no
  ancestry section at all rather than one row of nothing.

**Media reads as `media` in the attributes section, not `blob`.** It is stored
as a blob like anything else, but saying so sends a reader looking for an array.

**`links=True` for link contents was not built.** The section already gives the
kind, the count and what it carries, and the case for reading every link to show
its values did not survive having the names there. `verify=True` for media
digests is in, and off by default, because it re-reads every byte.

**The value cut uses three dots rather than an ellipsis character.** A
description is printed to whatever console is there, and a Windows one on a
legacy code page raises on `…` rather than showing it.

## Plan

1. ~~Fix `preview()` to exclude blobs by default and name what it excluded.~~
   Done — the omitted names are printed and recorded in
   `df.attrs['blobs_omitted']`, and `blobs=True` asks for them.
2. ~~A backend method for per-entity attribute metadata.~~ Done, plus the
   collection-level, link-name, link-count and child-count queries the sections
   needed: `get_entity_attribute_metadata`,
   `get_collection_attribute_metadata`, `get_link_attribute_names`,
   `count_collection_links_by_type`, `count_child_entities`,
   `count_collection_child_entities`.
3. ~~`Description`, its sections, and both reprs.~~
4. ~~`Entity.describe()`.~~
5. ~~`Collection.describe()`.~~
6. `verify=True` done; `links=True` not built, see above.

## The two follow-ups, built

### `Entarchy.describe()`

The question a stranger asks first, and the one nothing else answered: which
entity types are in here and how many of each, which link kinds join them, and
where the bytes went. Every other `describe()` needs you to know what to ask
about before you can ask.

    ent.describe()                  # renders whole
    ent.describe().entities         # a DataFrame
    ent.describe(largest=None)      # every attribute in the storage section

Three sections. **entities** is the census, in hierarchy order rather than
alphabetical — sorted by name it would put Roi above Recording and say nothing
about which contains which. **links** gives each kind, what it joins, its
cardinality, how many there are and what they cost. **storage** ranks the
attributes by size, largest first.

On the vxpy entarchy, **9.17 s** for 13.1 GB across 1.5 million attribute rows.
Everything but the storage section rides an index; totalling bytes cannot, so
that one scans the attributes table once, grouped down to a row per attribute
name per entity type. The cost is in how many rows there are, not how much data
they hold.

What it said there:

| | |
|---|---|
| `display/.../rotation_quats_0` | **4.9 GB** on 17 Recordings |
| `s2p/regPC` | 1.9 GB on 50 Layers |
| `fluorescence` | 763 MB across all 42,521 ROIs |

One stimulus attribute on 17 recordings is over a third of the entarchy.

A value is counted where it actually is, so a media file weighs what it weighs
on disk rather than what its pointer row weighs. `stored` is the size of the
entarchy, not of its database.

The entity census leaves link carriers out and the links section picks them up,
so the two together account for every byte exactly once — which is the reason
the census leaves them out rather than hiding them. A test asserts the identity.

### `distribution=True`

`Collection.describe(distribution=True)` adds `min`, `max` and `distinct`. Off
by default: it is a query per stored type on top of the one the section already
costs — **19.4 s against 4.8 s** on 42,521 ROIs — and a range is a question
about the data rather than about what is in there.

Endpoints are values, not renderings of them, so `row['max'] - row['min']`
works. Only text is cut, and only because `value_str` is a TEXT column.

The first thing it said on real data: **`s2p/footprint` is 1 on every one of
42,521 ROIs.** min 1, max 1, distinct 1. A column carrying no information at
all. Next to it, `s2p/aspect_ratio` and `s2p/radius` have 42,521 distinct values
and `s2p/compact` has 21,929 — half of them repeats.

**NaN and infinity had to be handled rather than inherited.** Both are stored as
a flag with a null value column, some dialects rejecting them outright, so
`MIN`, `MAX` and `COUNT(DISTINCT)` pass straight over them. A range that took
the query at its word would silently be a range over the finite values alone.
So infinity is folded back into the ends, NaN counts as one more distinct value
and stays out of the range, and a note names the attributes holding it.

The case that caught the first bug in this: an attribute whose *only* value is
`+inf`. Both `MIN` and `MAX` come back null, and my first version filled the
high end from the flag and left the low end with nothing to take a minimum of.
The archive fixture found it, because it already held a NaN and both infinities
from an earlier session.

**The second bug was worse, and only MySQL could show it.** Having asked the
database for a minimum and a maximum per stored type, I then took `min()` and
`max()` over all of them in Python to get one range per name. For a single type
that is a no-op on any value Python and the server order the same way — and
text is exactly where they do not. MySQL calls `abc` the least of
`['abc', 'ABC', 'Zeta', 'alpha']` and `Zeta` the greatest; handed just those
two, Python's byte order puts `Zeta` first. **The reported range came back the
wrong way round.**

It presented as a flaky test, which is what made it worth chasing rather than
re-running: under a case-insensitive collation `MIN` may return either `abc` or
`ABC`, and only the lowercase one loses to `Zeta` in byte order. So it failed
about half the time. One type's two ends are now kept as the pair the database
gave and never re-derived. Several types are still combined in Python, which is
safe because the only combinations Python will compare at all are the numeric
ones — text against anything raises `TypeError` and blanks the range, which is
the honest answer for a name stored two ways.

A unit test on `_range_of` pins it without needing a server: hand it
`min='abc', max='Zeta'` and both have to come back as they went in.

**Text ranges are the database's, not Python's.** `MIN`, `MAX` and
`COUNT(DISTINCT)` take the server's collation: SQLite compares bytes, MySQL 8
defaults to case- and accent-insensitive. `['abc', 'ABC', 'Zeta', 'alpha']` is
four distinct values on one and three on the other. Making them agree would mean
reading every value into Python, which is the cost this whole feature exists to
avoid — so they are allowed to differ, and a MySQL test pins that they differ in
the way the collations say rather than in some other way. Numbers, booleans and
times agree everywhere.

This is the same fact that made `Collection.sort()` order in Python. Sorting
could afford to; a distinct count cannot.

## Open questions

- Should `describe()` show a value *sample* for blobs — shape and dtype, which
  the container header already carries — or only type and size? Shape is what a
  reader usually wants, and reading a header is not reading a value, but it is
  one seek per blob rather than one query for all of them. Still the one
  unanswered question of the three.
- Should the storage section be able to group by entarchy *subtree* rather than
  by entity type — "what does this one animal cost"? The query is the same shape;
  it is the scoping that is not obvious.
