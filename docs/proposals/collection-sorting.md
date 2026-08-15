# Proposal: sorting entity collections

Status: **proposed**. Nothing built. The measurements below are of the current
code, not of an implementation.

Measured against: entarchy at `731d516`, SQLite 3.40.1, MySQL 8.0.46,
Windows 11 / Python 3.10.

## Summary

A collection has no order anyone asked for. `rois[0]` is whichever ROI happened
to sort first by UUID4, and `for roi in rois` walks them in that same arbitrary
order. Every place in the analysis code that needs a real order sorts in Python
after the fact:

```python
sorted(layer.rois, key=lambda r: r['index'])            # entarchy_vxpy_suite2p
sorted(names, key=lambda n: int(n.replace('plane','')))  # Suite2pSource.layer_names
```

This proposes `Collection.sort()`, returning a derived collection the way
`where()` and the set operators already do.

The recommendation is that **sorting happens in Python, on values fetched
through the pivot that `dataframe_of` already uses** — not as SQL `ORDER BY`.
That is the unusual half of this proposal, so the reasoning is set out before
the design. The short version: the two backends do not sort strings the same
way, and an entarchy's order should not depend on where it is stored.

## What order there is today

| path | order |
|---|---|
| `collection[i]`, `collection[a:b]` | `ORDER BY entities.uuid` |
| `for entity in collection` | same — the iterator takes a full slice |
| `map()` | same, it iterates |
| `map_async()` | uuid order, then **regrouped by parent** when `_locality=True` (the default) |
| `dataframe_of()` | **unordered** — the pivot has `GROUP BY` and no `ORDER BY` |
| `get_collection_parent_uuids()` | **unordered** — a bare `.all()` |

UUID4 order is arbitrary but stable, which is why nothing has broken. The last
two rows are the problem, and they are a problem *today*, before any of this is
built.

### The bug that had to be fixed first

**Fixed, ahead of the rest of this proposal.** It turned out not to be latent:
`dataframe_of` was returning parent attribute values attached to the wrong
entities, on SQLite, today. A ROI whose layer depth was 15.0 read as 0.0.

`dataframe_of` combined those last two paths **positionally**:

```python
parent_df = pd.DataFrame(index=self._cache.index)      # order of the pivot
uuids, parent_uuids = list(zip(*...get_collection_parent_uuids(self)))
...
parent_df[parent_attr] = parent_values                 # order of a different query
```

`self._cache.index` comes from the pivot's `GROUP BY entities.uuid, entities.id`.
`parent_values` is built by walking `parent_uuids` from a separate, unordered
query. The two were assumed to line up. Nothing made them, and they did not:
the pivot groups by uuid, while the parent query is a plain entity scan that
comes back in storage order. UUID4 keys are random, so the two orders are
unrelated and disagree almost always.

`uuids` was even unpacked on that line and then never used — it is exactly the
check that would have caught this.

The existing test passed throughout, because it asserted set membership rather
than per-row correctness:

```python
assert set(df['../depth']) == {0.0, 15.0}     # true however the rows are shuffled
```

The fix is to look the value up by the entity it is about:

```python
parent_of = dict(self.entarchy.backend.get_collection_parent_uuids(self))
...
for entity_uuid in self._cache.index:
    parent_uuid = parent_of.get(entity_uuid)
```

The correct pattern was already in the same file — `map_async` builds exactly
this mapping for its locality grouping. Only `dataframe_of` zipped.

## Why not `ORDER BY`

The obvious implementation is to push the sort into SQL. Four things argue
against it, one of them decisively.

### 1. The backends disagree about string order

Neither backend pins a collation, so each takes its server default. Measured on
the same fourteen strings:

```
SQLite  : ABC, Plane3, Roi_0, Roi_1, Roi_10, Roi_100, Roi_11, Roi_2, Zeta,
          abc, alpha, plane0, plane10, plane2
MySQL   : abc, ABC, alpha, plane0, plane10, plane2, Plane3, Roi_0, Roi_1,
          Roi_10, Roi_100, Roi_11, Roi_2, Zeta
```

SQLite compares bytes (`BINARY`): every uppercase letter sorts before every
lowercase one. MySQL 8 defaults to `utf8mb4_0900_ai_ci`, which is case- and
accent-insensitive. These are not small differences — `Zeta` is second-to-last
in one and last in the other, and `abc`/`ABC` swap.

For a tool whose purpose is to make an analysis reproducible, "the tenth ROI"
meaning two different ROIs depending on the backend is not acceptable. And it is
worse than a deployment detail: **`ArchiveBackend` subclasses `SQLiteBackend`**,
so an archive exported from a MySQL entarchy would sort differently from the
entarchy it was exported from. An archive is supposed to be the citable, frozen
artefact.

Python's `sorted()` gives byte order, matching SQLite exactly. So a Python-side
sort changes nothing for SQLite entarchies and archives, and brings MySQL into
line with them rather than the reverse.

### 2. `NULLS LAST` is not portable

Attributes are per entity, not per type, so an entity may simply not have the
sort key — the outer join gives NULL. Both backends put NULL first ascending.
Putting the missing ones last is usually what a reader wants, and:

```
MySQL supports NULLS LAST: no (ProgrammingError)
```

SQLite has had it since 3.30; MySQL 8 has never had it and needs an
`ISNULL(x), x` prefix column instead. Two dialects, hand-written, for something
`pandas` expresses as `na_position='last'`.

### 3. The sort key lives in an EAV pivot anyway

There is no column to order by. The value is in one of seven typed columns
picked by `data_type`, so `ORDER BY` needs the same `MAX(CASE …)` construction
the read path builds — a join and a group per sort key, with no index on any
value column, so every sort is a filesort regardless. The work is comparable to
loading the column and sorting it, which is what the pivot already does.

### 4. Sort-then-slice would be quadratic through `__getitem__`

`collection[i]` issues one query per index. Today that is an indexed `OFFSET` on
the primary key. With a sort over a join it becomes a full sort per call, so a
`for i in range(len(c))` loop sorts the collection `len(c)` times.

### What is given up

Push-down. `ORDER BY … LIMIT 10` lets the database stop early; sorting in Python
means fetching every key to find the top ten. That is real, and it is the reason
to revisit this if entarchies grow by an order of magnitude. At the sizes this
is built for — 42,521 ROIs in the entarchy just ingested — one pivot of one
column is a query `dataframe_of` already runs routinely.

If push-down is ever wanted, it can be added underneath as an optimisation for
the narrow case where it is safe (numeric key, no ties, one backend), without
changing the API. Doing it the other way round — starting with `ORDER BY` and
later discovering the archive sorts differently — is not recoverable.

## The design

Sorting is state on the collection, alongside `_as_tree`, and `_derive` carries
it. Lazy: `.sort()` runs no query.

```python
rois.sort('index')                      # ascending
rois.sort('-dff_max')                   # descending
rois.sort('layer_index', '-dff_max')    # major, then minor
rois.sort('id', natural=True)           # Roi_2 before Roi_10
rois.sort('snr', missing='first')       # default is 'last'
```

Resolution order, applied once when the collection is first materialised:

1. Fetch the key columns via the existing pivot (`dataframe_of(keys)`).
2. Sort with pandas, `na_position` from `missing`.
3. **Always append uuid as the final key**, so ties are broken deterministically.
4. Keep the resulting uuid order on the collection; every access path reads it.

Step 3 is not a detail. Without it, "the first ten ROIs by response" can change
membership between runs whenever the planner changes, which is the sort of
irreproducibility that is very hard to notice and very annoying to explain.

### What each access path does with it

- `__getitem__`, slices, `__iter__`, `map()` — index into the stored uuid order,
  then fetch by uuid rather than by `OFFSET`.
- `dataframe_of` — reindex the cache. The author already anticipated this; the
  comment is still in the file:
  ```python
  # TODO: return final DataFrame in custom order
  # if self._query_custom_orderby:
  #     return self._cache.loc[self._pk_order, attribute_names]
  ```
- `map_async` — see the pitfall below.

### Sorting by parent attributes

`[Animal]id` and `../rate` work in filters and in `dataframe_of`, so readers will
expect them here. They come free: `dataframe_of` already resolves them, and the
sort reads its output. Worth stating as supported rather than leaving it to be
discovered — it is one of the few places where doing this in Python is *simpler*
than the SQL would have been.

## Pitfalls

Ordered by how much trouble each will cause.

**1. The positional join in `dataframe_of` (above).** Done, ahead of the rest —
it was not latent but actively wrong. Four tests cover it: three that put the
lookup out of order deliberately, and one that simply asks whether the frame
agrees with the entities, which is the one that would have caught it years ago.

**2. Lexicographic order on ids is not what anyone means.** This schema names
things `Roi_0 … Roi_1299` and `plane0 … plane4`, so plain sorting gives
`Roi_1, Roi_10, Roi_100, Roi_11, Roi_2`. The codebase has already been bitten:
`Suite2pSource.layer_names` carries `key=lambda n: int(n.replace('plane',''))`
precisely because `sorted()` was wrong. A `sort('id')` that reads as "in order"
and is not will be reported as a bug. Hence `natural=True` — and there is a case
for making it the *default* for `id`, since no caller has ever wanted the
lexicographic answer.

**3. Blob attributes cannot be sort keys.** `fluorescence` is an opaque encoded
container by design. Ordering by it would compare container bytes — a magic
number, then a header — which is meaningless and would look like it worked.
`sort()` must look up `data_type` and refuse, naming the attribute.

**4. An attribute's type is decided globally, not per collection.** The pivot
asks the whole table what `data_type` a name has (deliberately — it was the
single largest cost of `get_collection_attributes`). Where one name is `int` for
some entities and `float` for others, one column wins and the rest read NULL.
Sorting inherits that silently, and a key that is mostly NULL for a bad reason
looks exactly like a key that is mostly missing for a good one. Worth a warning
when a sort key resolves to NULL for a large fraction of the collection.

**5. `map_async(_locality=True)` reorders, and it is the default.** It groups
entities by parent so a worker reuses a parent's cached attributes — the
docstring says so. Sorting and then calling `map_async` would not process in
sorted order. Options: have `_locality` default to `False` when a sort is set,
or refuse the combination, or document it loudly. Silence is the wrong choice;
today it does not matter because nobody asked for an order, and the moment
sorting exists someone will.

**6. Set operations have to say what they do.** `sorted_a | sorted_b` has no
obvious meaning. `_derive` builds a fresh collection from a combined tree, so
sort state is dropped unless carried. Simplest defensible rule: **set operations
drop the sort**, and the caller re-sorts the result. Document it; do not let it
be an accident of implementation.

**7. Cache coherence.** `_cache` is uuid-indexed, so reindexing is safe — but
`_length` and any stored uuid order must be invalidated together, and a sorted
collection that is written to (`collection['x'] = series`) must not reorder the
series against the cache. `__setitem__` builds a frame on `self.index`, so it
follows the cache; that stays correct only if the sort is applied at read time
rather than baked into `_cache`.

**8. Cost is now paid up front.** `len(c)` is a `COUNT`; today `c[0]` is one
indexed row. Sorted, the first access fetches every key in the collection. On
42k ROIs that is one pivot column — fine. It should still be lazy, so a
collection that is sorted and then never read costs nothing, and it should be
obvious in the docstring that the first read is the expensive one.

**9. NaN is not missing.** Floats store NaN/Inf as NULL plus a marker column, so
a missing attribute and a NaN value both arrive as NULL from the pivot and are
told apart by `float_is_nan` / `float_is_inf`. `missing='last'` must not quietly
merge the two — an ROI with no `snr` and an ROI whose `snr` is NaN are different
facts, and the pivot already returns enough to distinguish them.

**10. Ties in float keys.** Only worth noting because step 3 handles it. Without
the uuid tiebreaker, `sort('dff_max')` over ROIs where many share a rounded
value gives an order that depends on pandas' sort kind. `kind='stable'` plus the
uuid key makes it reproducible.

## Plan

1. Fix the positional join in `dataframe_of`, with a regression test. Independent
   of the rest.
2. Give the pivot and `get_collection_parent_uuids` an explicit `ORDER BY uuid`,
   so the unsorted case is specified rather than incidental.
3. `Collection.sort()` — state, `_derive` carrying it, resolution through the
   pivot, uuid tiebreaker.
4. Wire the read paths: `__getitem__`, slices, `__iter__`, `dataframe_of`.
5. Decide and implement the `map_async` interaction; decide the set-operation
   rule.
6. `natural=` and the blob refusal.

Steps 1 and 2 are worth doing on their own merits even if sorting is dropped.

## Open questions

- Should `natural=True` be the default for `id`? It is what every caller has
  wanted so far, but a default that inspects the key's contents is a default
  that sometimes surprises.
- Should `sort()` on an unsorted-by-default collection make the *implicit* uuid
  order explicit in the repr, so `preview()` shows what one is looking at?
- Is there a case for `sort_by_hierarchy()` — parent id, then child id, down the
  tree? It is what most printed output wants, and it is tedious to spell out
  as `sort('[Animal]id', '[Recording]id', 'id')`.
