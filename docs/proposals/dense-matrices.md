# Proposal: pairwise values that are dense

Status: **proposed**. Nothing implemented.

Measured against: entarchy at `8463045`, on the vxpy entarchy at
`E:/data/entarchy_vxpy` — 38 recordings, 17,079 phases, 42,521 ROIs. SQLite
3.40.1, Windows 11 / Python 3.10.

Revised 2026-11-19 against `c453e43`. The storage layer the measurements rest on
— `blob_store`, the attribute tables, the link tables — has not changed since
`8463045`, so the numbers stand. What changed is the API around them: how
entities are built and added, how a subtree is queried, and what an analysis
entity is. The text below is brought in line, and the differences are listed at
the end.

## Summary

There are two ways to hold a value per pair of entities, and entarchy has an API
for one of them.

|  | per pair | queryable | API |
|---|---|---|---|
| links | **1,054 B** | yes, from either end, in SQL | `link_from_matrix` / `matrix_from_links` |
| an array attribute | **3.6 B** | no | none |

Both measured, on one float per pair. The array is **290× smaller** and there is
no way to ask for one: you write the array, you write the axis labels beside it
under a convention of your own, and you hope the two stay together. Lose the
labels and the array is positional, which is to say correct until somebody
re-sorts a collection.

`matrix_from_links` already returns the right *shape* of answer — a DataFrame
indexed by the uuids of two collections. This proposes the same answer for data
that should never have become links.

## Where 1,054 bytes goes

The same links carrying 0, 1 and 2 floats, measured on disk:

| | bytes per link |
|---|---|
| a link carrying **nothing** | **839** |
| carrying one float | 1,054 (+216) |
| carrying two floats | 1,266 (+212) |
| the same float in an array | **3.6** |

**Eighty percent of the cost is the carrier, before any data exists.** A link is
an entity row, a link row and four indexes; between them they hold the two
endpoint uuids about sixteen times over as 36-character text. That is what buys
addressability from either end, and for a dense result nobody addresses a cell.

The case that prompted this: `Phase -> Roi` mean responses in the CMN analysis.
16,955,713 real pairs — **16.6 GB as links, 65 MB as float32 arrays.**

## Storage: one attribute, verified

`blob_store` already encodes a dict of arrays natively. A float32 matrix, two
`<U36` label arrays and a couple of scalars round-trip through one attribute
unchanged:

```python
entity['mean_dff'] = {'values': ..., 'rows': ..., 'columns': ...}
```

450 × 1119 came back at **1.74 MB on disk, 3.63 B per cell** — under raw float32,
because the label arrays compress. **No backend change, no new table, no
migration.** This is a convenience layer over storage that already works.

The labels are the point. An array alone is positional; the uuids are what make
it survive a re-sort, a filtered subset or a read in a different order — the same
reason `matrix_from_links` reindexes rather than trusting position.

## Ownership: the analysis, not the hierarchy

An earlier draft of this proposal put the matrix on the **lowest common ancestor**
of its two axes. That was wrong, and the reason it was wrong is worth writing
down.

For `Phase × Roi` within one recording the LCA is that recording, which is fine.
For anything spanning recordings — every ROI against every ROI, every ROI against
a shared stimulus geometry — the LCA is the entarchy root. And the root is the one
place a matrix must not go:

| one matrix over every phase × every roi | |
|---|---|
| cells | 726,216,159 |
| cells that can mean anything | 16,955,713 — **2.3%** |
| as float32 | 2.71 GB against 65 MB block-diagonal |

So the rule collapses to "the root" in exactly the cases where the root is the
worst answer. Deriving the owner also means the author cannot predict where two
megabytes went, and `Entarchy.describe()` reports it under a type nobody chose.

**`AnalysisEntity` is the right owner instead**, and it already has most of the
properties the job needs:

- **Parentless, and outside the hierarchy.** `set_current_analysis('cmn_rf')`
  creates the analysis entity with `parent is None` — verified against `c453e43`.
  Since the hierarchy became authoritative and grows in the store, the built-in
  types are not part of it at all: `ent.add_child(AnalysisEntity, 'x')` is refused
  by the same type check every other write goes through, and the only way in is
  `set_current_analysis`. Global is its *correct* scope, not a fallback it
  degraded into, and nothing a user grows into the hierarchy can change that.
- **Named by the author.** `set_current_analysis('cmn_rf')` is idempotent and
  reuses the stored uuid across sessions. Nothing is derived or guessed.
- **Already a context manager.** `with analysis:` sets it current and restores the
  previous one on exit, so writes already have an ambient owner.
- **Segregation for free.** The attributes primary key is `(entity_uuid, name)`,
  so two analyses each holding `mean_dff` are two rows on two entities. No new
  mechanism, no stamp to honour. The class docstring now says this in so many
  words.
- **Not yet the deletion unit.** An earlier revision claimed a derived result goes
  away with `delete()` on the analysis. There is no such method: the only
  `delete()` is `Entarchy.delete()`, which removes the whole store after an
  interactive confirmation, and no entity-level removal exists on `Entity`, the
  `Entarchy` or the backend interface. The lifetime argument for putting matrices
  on the analysis is still right; the mechanism that would honour it is a gap,
  listed under the open questions.

And conceptually it is simply true: a pairwise result is a fact about a
computation, not about a recording. Putting it on the recording says the
recording has a `mean_dff`, which it does not — some analysis of it does.

### What this turned up: `analysis_uuid` did not segregate

Every attribute row used to carry an `analysis_uuid`, and `AnalysisEntity`'s
docstring was "Analysis entity for data segregation". Measured:

- Two analyses wrote `recording['result']` in turn. **One row existed**, holding
  the second value and **stamped with the first analysis's uuid.** The primary key
  is `(entity_uuid, name)`, so the second write overwrote the first, and the stamp
  was not updated.
- The column was written in two places in `sql.py` and **read in none.** No query
  filtered on it, and on the vxpy entarchy 0 of 1,625,093 rows carried a value.

So it recorded who created a row, went wrong as soon as anyone else touched it,
and was never consulted. **It was dropped in `7eabd01`**, which leaves the point
that matters here standing: **build on the entity, not on a stamp.** A matrix
stored *on* the analysis entity is segregated by the primary key, which works.

## The API

```python
ent.set_current_analysis('cmn_rf')

with ent.current_analysis as analysis:
    analysis.set_matrix('mean_dff', values, rows=phases, columns=rois)

analysis.matrix_of('mean_dff')                      # as stored
analysis.matrix_of('mean_dff', rows=good_rois)      # reindexed, subset
rois.matrix_with(phases, 'mean_dff')                # from the current analysis
```

All three return a DataFrame indexed by uuid with the axes named for their entity
types — identical in shape and convention to `matrix_from_links`, so moving a
quantity between the two storage forms does not change the code that reads it.

`matrix_with` reads from `ent.current_analysis` and raises if none is set rather
than guessing. Writes go to the current analysis, reads come from it: the ambient
scoping `analysis_uuid` was reaching for, done where it works.

### `matrix_with` versus `matrix_of`

The collection form is the better-looking call and should exist: the two
collections are the actual arguments, and it matches `links_to` and
`matrix_from_links`, so a reader who has met one has met all three.

It cannot be the only form. `a.links_to(b, 'kind')` works because links exist
independently of `a` and `b` — the collections *select* among things already
stored. A matrix does not exist independently; it is a blob, and a blob sits on an
entity. `rois.set_matrix_with(phases, ...)` would have to choose an analysis, a
name and a scope on the caller's behalf. Writing names its owner; reading does not
have to.

### Several matrices under one name

Per-recording matrices are 38 separate arrays, and entarchy already has a
convention for that — `/`-separated names, as in `s2p/npix` and `ants/x`. The
axes come from the recording itself: `Entity.get` is `ent.get` scoped to an
entity's subtree, at any depth, so no schema-level property is needed to name
them.

```python
for recording in ent.get(Recording):
    analysis.set_matrix(f'mean_dff/{recording.id}', values,
                        rows=recording.get(Phase), columns=recording.get(Roi))
```

Each block is independently readable, which also answers "reading reads all of
it": you open the block you want, not the set. Assembling blocks automatically is
deliberately **not** in this proposal — see the open questions.

## Pitfalls

**Rewriting one cell rewrites the blob.** The real price of the array form, and it
belongs in the docstring rather than being discovered. Links let you update one
pair; a matrix does not. A result built incrementally wants links, or wants
assembling in memory and writing once.

**Reading reads all of it.** `blob_store` has no partial read, so a 42,521 ×
42,521 correlation would pull 7 GB. Links stream; this does not. A `rows=` subset
argument still reads the whole blob first and is honest only if it says so.

**Staleness is visible, not prevented.** The labels name uuids. Entities added
after the write come back as NaN rows, which is correct and quiet. Nothing checks
that a matrix predates half its axis.

**Two analyses, one name, one entity each.** This is the case the primary key
handles — but only because the matrix is on the analysis. Anyone who instead
writes it onto a `Recording` from two analyses gets the silent overwrite described
above.

**Symmetric matrices are simpler here than in links.** Both axes are stored
explicitly, so there is no uuid-ordering artifact and no mirroring — the whole
class of orientation bug `matrix_from_links` has to handle does not arise.

**`describe()` will call it a blob.** One attribute holding a dict is a blob like
any other. Either a `matrices` section, or at minimum a type reading `matrix` the
way media reads `media` rather than `blob`.

## Plan

1. `AnalysisEntity.set_matrix(name, values, rows, columns)` — capture the labels,
   write one attribute. Refuse mismatched shapes and a name already holding a
   non-matrix.
2. `AnalysisEntity.matrix_of(name, rows=None, columns=None)` — read, reindex, name
   the axes, following the `matrix_from_links` return convention exactly.
3. `Collection.matrix_with(other, name)` — resolve the current analysis, delegate.
   Raise, rather than guess, when none is set.
4. `describe()` reports matrices as matrices rather than as anonymous blobs.
5. Round-trip tests against SQLite, MySQL and an archive. The storage is an
   ordinary blob so an archive should need nothing special — a claim worth a test
   rather than an assumption.

Steps 1–3 are `core/` only. Nothing touches SQL.

## Open questions

- **Should `set_matrix` and `matrix_of` be on `Entity` rather than
  `AnalysisEntity`?** Putting them on `Entity` costs nothing and lets a schema
  hang a matrix on a `Recording` deliberately. It also makes the silent-overwrite
  trap reachable. The API has since moved toward one verb at every level —
  `add_child` on the entarchy and on any entity, `get` on the entarchy and on any
  entity — which is an argument for `Entity`; the trap is an argument against.
  Narrow first is the safer order.
- **Deleting an analysis.** No entity can be removed today, so "a derived result
  goes away with the analysis that derived it" is a design intent without a
  method. An `AnalysisEntity.delete()` that removes the entity and its attribute
  rows would make it true, and is the first removal entarchy would have, so it
  deserves its own proposal rather than a line in this one.
- **Automatic block assembly.** `rois.matrix_with(phases, 'mean_dff')` across 38
  recordings could find and stitch the blocks, but that needs an index from entity
  uuid to block — a small extra attribute per name, since the labels live inside
  the blobs. Worth building only once someone wants the stitched frame rather than
  a loop over recordings.
- **Axes that are not entities.** The CMN patch geometry is 320 patches that have
  no entity and may never deserve one. A matrix whose axis is labelled by an
  integer index or an attribute value, rather than by uuid, would cover it — and
  would make the visual-field involvement map a matrix rather than six duplicated
  unpacking loops.
- **Is `matrix_with` the right name next to `matrix_from_links`?** The pair does
  not say that one reads links and the other an array — arguably correct, since a
  caller should not care, and arguably a trap for the one who does.

## Changes in entarchy since `8463045`, and what they mean here

- `c365ee5`, `353268f` — entities are built by id alone, `Subject('id')`, and
  bound when `add_child` takes them; `add_new_entity` is gone and `_add_entity`
  is private. The constructor form this proposal used for the analysis entity,
  `AnalysisEntity(ent, _id='cmn_rf')`, no longer exists and was never the user
  path: `set_current_analysis` is.
- `62d0746`, `92422bd` — the stored type hierarchy is authoritative and grows
  with `add_entity_type` and `add_child_entity_type`; the built-in types sit
  outside it. An analysis entity cannot end up under a user type by accident.
- `c453e43` — `Entity.get(Type, ...)` queries an entity's subtree. The
  per-recording example uses it instead of schema properties.
- `d03bbc0` — `Entity.where(...)` replaced the `Entity('<query>')` form. Not used
  here, but any example that filtered an axis that way would now read
  `Roi.where(...)`.
- `7eabd01` — the `analysis_uuid` stamp was dropped, as described above.
- `3a3b064`, `dc8af47` — `create()` defaults to SQLite in the folder and
  `open_or_create()` exists. Irrelevant to the proposal; noted because the
  examples' setup lines would be shorter today.
