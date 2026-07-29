# Proposal: an ASDF-based storage layer for entarchy

Status: **Option C implemented** (`entarchy.tools.archive`,
`entarchy.backend.archive`, `entarchy.backend.asdf_store`). Options A and B
remain proposals; see "What was built" below for how the implementation differs
from what this document originally described.
Measured against: asdf 5.3.1, ASDF Standard 1.6.0, numpy 2.x, Windows 11 / Python 3.10

## Summary

ASDF is a very good fit for the *array* half of entarchy's data and a poor fit for
the *entity* half. Its binary blocks are memory-mappable, self-describing and
archival; its metadata tree is a YAML document that has to be parsed in full on
every open, which does not scale to the entity counts entarchy targets.

The recommendation is therefore **not** to add a third backend beside SQLite and
MySQL, but to add ASDF as a **blob storage layer underneath the existing
backends**, replacing the current opaque `ext/` shard files, plus a **standalone
export** for archiving and sharing. Both are additive and neither disturbs the
query path.

A full ASDF backend is described in Option A below and measured; it is not
recommended, and the numbers explaining why are in the next section.

## What ASDF is

The Advanced Scientific Data Format, developed at STScI and used as the native
format of JWST. A file is a human-readable YAML 1.1 tree followed by binary
blocks:

```
#ASDF 1.0.0
#ASDF_STANDARD 1.6.0
...tree in YAML...
...binary blocks...
```

Relevant properties, all verified against 5.3.1:

- numpy arrays are stored as binary blocks and returned memory-mapped, so reading
  one array from a large file does not read the rest.
- Arrays can live in the same file or in external files
  (`all_array_storage='external'`, which leaves an 781-byte tree file plus one
  file per array).
- Per-array compression (`all_array_compression='zlib'` and others). A 1.60 MB
  array of zeros became 2.3 kB.
- Schemas are versioned, and custom types can be given schemas and converters, so
  a file records what its contents *mean*, not just their shape.
- Single writer. There is no locking, no transaction, and no query language.

## Measured characteristics

Tree of *n* entries, each with two scalars and a 200-element array:

| entries | write | file size | lazy open | read one array |
|--------:|------:|----------:|----------:|---------------:|
| 500 | 0.48 s | 0.9 MB | 0.26 s | 0.34 ms |
| 2 000 | 1.12 s | 3.6 MB | 1.02 s | 0.35 ms |
| 8 000 | 4.70 s | 14.4 MB | 4.44 s | 0.39 ms |
| 20 000 | 12.89 s | 36.1 MB | 11.99 s | 0.39 ms |

Two things stand out.

**Array access is flat and fast.** 0.4 ms regardless of file size, because blocks
are memory-mapped and indexed by the tree. This is exactly what the CMN analysis
needs when it reads one ROI's trace or a layer's motion vectors.

**Opening scales linearly with the number of entries.** Roughly 0.6 ms per entry,
because the whole YAML tree is parsed before anything can be addressed.
Extrapolated to the 100 000 ROIs entarchy is built for, that is **about 60
seconds per open** — paid again in every one of six `map_async` worker processes,
against a few milliseconds for the equivalent indexed SQL lookup. Scalar access
*after* opening is free (a scan of 2 000 scalar fields took 0.3 ms), so the cost
is entirely in the open.

## Fit against the Backend interface

`Backend` has 20 methods. Grouped by how well ASDF serves them:

**Natural fit** — `get_entity_attribute(s)`, `set_entity_attribute(s)`,
`has_entity_attribute`, `get_entity_attribute_names`, `get_entity_parent`,
`get_entity_by_uuid`. These are keyed lookups; a tree indexed by UUID handles them
directly, and array attributes come back memory-mapped.

**Awkward** — `add_entities`, `create_type_hierarchy`. Writing means rewriting or
appending to the tree. In-place update works (`asdf.open(mode='rw')` then
`update()`), but the file only grows: adding one 1000-element array to a
8.8 kB file took it to 16.9 kB, and freed space is not reclaimed. Bulk ingest
would need to build the tree in memory and write once.

**Poor fit** — everything collection-shaped: `get_collection_count`,
`get_collection_entities_by_slice`, `get_collection_attributes`,
`set_collection_attributes`, `get_collection_parent_uuids`. entarchy turns a
filter expression into SQL and lets the database use its indexes. ASDF has no
query engine, so every one of these becomes a full Python traversal of the tree,
after the full-tree parse measured above.

**No equivalent** — concurrent writers. `map_async` runs six worker processes that
all commit attributes. SQLite serializes them with its own locking and MySQL is
built for it; ASDF is a single-writer format with no locking API. This alone rules
out a drop-in backend for the current parallel write pattern.

## Options

### Option A — a full ASDF backend beside SQLite and MySQL

Implement all 20 `Backend` methods against one ASDF file (or one per animal).

Attractive because a dataset becomes a single self-describing file with no server
and no schema migrations, and because the format is archival in a way neither of
the current backends is.

Not recommended, for three reasons, in order of severity:

1. **Concurrent writes.** Six workers writing attributes to one ASDF file has no
   safe implementation without inventing a locking protocol on top.
2. **Query cost.** Every filter becomes a full traversal, after a full-tree parse:
   ~60 s at 100 000 ROIs versus milliseconds today.
3. **Update amplification.** Analysis writes derived attributes constantly; each
   one grows the file, and the file is rewritten to reclaim space.

Options B and C keep every advantage that motivated the idea and avoid all three.

### Option B — ASDF as the blob store (recommended)

entarchy already writes large attributes outside the database: `Serializer`
spills anything over `max_blob_size` into `ext/<uuid shards>/<sha224>.npy` or
`.pickle`, and stores a pickled `Serializer` in the `value_blob` column pointing
at it.

Two things are wrong with that layer today, independent of ASDF:

- The files are opaque. `ext/01fb/4222/.../8a3f2c….npy` says nothing about which
  entity or attribute it belongs to; the mapping lives only in the database.
- Non-array values go through `pickle`, which is neither portable nor safe to
  read from an untrusted source, and which nothing outside Python can open.

Replacing that layer with ASDF fixes both without touching the query path:

- One ASDF file per entity (or per parent group), holding that entity's large
  attributes as named blocks, with the entity UUID, its id, its path in the
  hierarchy and the attribute names in the tree.
- The database keeps a pointer, exactly as it does now, so all querying is
  unchanged.
- Arrays come back memory-mapped rather than read whole, which directly helps
  the case measured earlier: a layer's `cmn_motion_vectors_2d` is roughly 120 MB
  and is currently deserialized in full by every worker that touches that layer.
- Per-array compression becomes available for free.
- An `ext/` tree becomes self-describing: each file states what it contains, so a
  dataset survives losing its database.

The `Serializer` abstraction already isolates this. The change is a new storage
strategy inside it plus a `store` marker distinguishing `'asdf'` from the current
`'internal'` and path forms, with the existing readers kept for old data.

### Option C — ASDF as an export and archive format

A `to_asdf(path)` on `Entarchy` or `Collection` writing a subtree — an animal, a
recording, a filtered collection — as one self-describing file, with the entity
hierarchy in the tree and the arrays as blocks.

This is where ASDF's provenance and schema versioning genuinely pay: a file that
can be attached to a paper, deposited in a repository, and opened in ten years
without entarchy, a database server, or the pickle classes.

Cheap to build, independent of Options A and B, and useful immediately.

As first described, this had a defect: an archive only `asdf.open()` could read
is useless to the analysis and figure code, which is written against entarchy's
API. The implementation therefore makes the archive *an entarchy* rather than a
foreign format — see below.

## Recommended plan

1. **Option C first.** Small, self-contained, no risk to existing data, and it
   makes the archival argument concrete. It also builds the tree-mapping code
   that Option B reuses.
2. **Option B next**, behind a config switch (`blob_storage: asdf`), with the
   existing `ext/` readers retained so old entarchies keep working. Migration is a
   background rewrite, not a breaking change.
3. **Option A only if** the usage pattern changes to single-writer, read-mostly
   datasets under roughly 10 000 entities, where the open cost is tolerable.

## What was built

An archive is a normal entarchy directory whose backend happens to be
`ArchiveBackend`:

    archive/
        entarchy.yaml     names entarchy.backend.archive.ArchiveBackend
        index.sqlite      queryable metadata, normal entarchy schema
        meta.asdf         the same metadata, columnar and self-describing
        blocks/*.asdf     the arrays, one file per parent group

`Entarchy(path)` resolves its backend from a dotted path in `entarchy.yaml`, so
this needs no change anywhere above the backend layer: queries, filter
expressions, DataFrames, parent traversal and `map_async` all work against an
archive as written. `ArchiveBackend` subclasses `SQLiteBackend`, points the
engine at `index.sqlite` (opened `mode=ro` where the driver allows it) and
rejects the eight write methods.

None of the three objections that ruled out Option A apply, because the target
is read-only: there are no concurrent writers, no updates, and queries never
touch ASDF.

Three decisions worth recording, each forced by a measurement rather than taste.

**Metadata is columnar.** One YAML node per attribute row would reintroduce the
full-tree parse this document argues against. `meta.asdf` stores one array per
column plus a null mask, so a dataset of any size costs a fixed handful of
blocks. An outside reader gets clean columns; `rebuild` reconstructs
`index.sqlite` from it, which is what makes the index a cache rather than a
second source of truth.

**Ragged collections are packed.** The original note above was wrong to describe
`bs_cluster_full_indices` as needing a schema: it is a list, and ASDF handles
plain Python containers natively. The real problem is that a list of arrays
becomes one binary *block* per array. Measured on one ROI's attribute — 1000
bootstrap iterations, each a list of index arrays:

| encoding | blocks | size | write | read |
|---|---:|---:|---:|---:|
| naive (list of arrays) | 2 967 | 1 212 kB | 1 726 ms | 1 451 ms |
| concatenated + offsets | 3 | 767 kB | 39 ms | 29 ms |
| pickle (what it replaces) | — | 836 kB | 13 ms | 8 ms |

`asdf_store` concatenates such lists and keeps their boundaries in an offsets
array. Values come back bit-identical. Pickle is still the fastest of the three;
ASDF buys portability and self-description, not speed.

**Tuples and bytes are tagged.** YAML has neither. A tuple round-trips to a list
silently, which is exactly the kind of quiet type drift that breaks analysis code
far from the cause, so both are marked explicitly and restored on read.

Two things that were not obvious in advance:

- asdf returns `NDArrayType` proxies, not `ndarray`. They behave like arrays
  arithmetically but fail `isinstance`, and pickling one raises
  `TypeError: cannot pickle 'weakref.ReferenceType' object` — which is what
  entarchy's own `Serializer` does when it does not recognise a value as an
  array. Decoded values are materialised into real arrays.
- Block files are opened per process and cached, since opening costs roughly
  0.4 ms per tree entry (`lazy_tree=True` helps by about a third, not enough to
  change the design) against 0.3 ms for the array read itself. Memory mapping is
  off by default: a mapped array stops being readable once its file is closed,
  and cached files are closed on eviction.
- That cache has to be bigger than the number of groups a read pass touches, or
  every read evicts a file the next one wants. Reading 600 ROIs spread over 12
  groups: 1.31 ms per entity from a live entarchy, 1.93 ms from an archive with
  room for all 12, and 16.94 ms with room for 8. The default is 64 and
  `ArchiveBackend(open_file_limit=...)` raises it. So an archive costs about 1.5x
  a live entarchy per attribute read, not the 13x a too-small cache produced.

Answering the granularity question this document left open: **one file per parent
group**, matching the locality grouping `map_async` uses. One file per entity
makes 100 000 files; one file for everything makes every array read pay a
full-tree parse.

## Open questions

- **Compression policy for the array blocks.** Off by default is the safe choice,
  since imaging traces compress poorly and CPU is the bottleneck.
  `--compression zlib` is available and untested against real recordings.

  Settled for the *metadata*, which is always compressed. The columns are fixed
  width numpy strings, so every row is padded to the longest value in its column
  and one long `value_str` widens all of it; a 36 byte uuid becomes 144 in UCS4.
  That padding is pure redundancy: a 1994 attribute export went from 2185 kB to
  65 kB. At 3 million attribute rows it is the difference between roughly 3 GB
  and 100 MB, and `meta.asdf` is only read when the index is rebuilt.
- **Whether Option B is still wanted.** The archive path already contains the
  encoder and the `asdf:` store form, so the remaining work for a live ASDF blob
  layer is the write side plus a migration. The case for it is weaker now that
  archives cover the portability argument.
- **Dependency weight.** `asdf` pulls in `jmespath`, `semantic-version` and
  `attrs`. It is an optional extra (`pip install entarchy[asdf]`), and everything
  else works without it.

## What I would need from you

- Whether anything currently reads the `ext/` tree directly, which would
  constrain how far the layout can change under Option B.
- Whether real datasets hit the pickle fallback often. The exporter lists every
  attribute it could not encode natively; if that list is long on your data,
  those attributes are worth reshaping at the point they are written.
