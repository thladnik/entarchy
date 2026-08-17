# Proposal: seeing what an entity or collection holds

Status: **proposed**. Nothing built. The measurements are of the current code.

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
description.links        # kind, direction, other end, attributes of the link
description.media        # name, media type, bytes, present
description.children     # type, count
description.ancestry     # type, id  — up to the root

description              # renders whole, in a notebook
print(description)       # renders whole, in a terminal
```

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
thousands. `describe()` must cap what it lists and say that it capped —
`link_counts()` is already the cheap total, so the counts can be exact while the
listing is not. Silently showing the first ten of four thousand would be worse
than showing none.

**3. Reading a link's attributes means reading link entities.** The counts come
from an indexed `GROUP BY`; the *contents* do not. If `description.links` shows
link attributes it is doing real work per link, and that has to be opt-in
(`roi.describe(links=True)`) rather than the default for an entity that might
have thousands.

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

## Plan

1. Fix `preview()` to exclude blobs by default and name what it excluded.
   Independent of everything else, and the current behaviour is a trap.
2. A backend method for per-entity attribute metadata — name, type, size — in
   one query. `get_attribute_data_types` is the collection-level half of this
   and can be extended rather than duplicated.
3. `Description`, its sections, and both reprs.
4. `Entity.describe()` — attributes, media, ancestry, children, link counts.
5. `Collection.describe()` — the same, aggregated, with the `entities` coverage
   column.
6. Opt-in depth: `links=True` for link contents, `verify=True` for media
   digests.

Step 1 is worth doing on its own. Steps 2–5 are the proposal proper.

## Open questions

- Should `describe()` show a value *sample* for blobs — shape and dtype, which
  the container header already carries — or only type and size? Shape is what a
  reader usually wants, and reading a header is not reading a value, but it is
  one seek per blob rather than one query for all of them.
- Should `Collection.describe()` report the *distribution* of a scalar attribute
  (min/max/distinct count)? It is genuinely useful and it is genuinely
  `describe()` in the pandas sense, which cuts both ways for the naming.
- Should there be an entarchy-level `describe()` — entity counts per type, link
  kinds, total size, where the blobs are? That is the question "what is in this
  entarchy at all", which is the one a stranger to it asks first.
