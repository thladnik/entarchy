# Proposal: an ASDF-based storage layer for entarchy

Status: draft for discussion
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

## Recommended plan

1. **Option C first.** Small, self-contained, no risk to existing data, and it
   makes the archival argument concrete. It also builds the tree-mapping code
   that Option B reuses.
2. **Option B next**, behind a config switch (`blob_storage: asdf`), with the
   existing `ext/` readers retained so old entarchies keep working. Migration is a
   background rewrite, not a breaking change.
3. **Option A only if** the usage pattern changes to single-writer, read-mostly
   datasets under roughly 10 000 entities, where the open cost is tolerable.

## Open questions

- **Granularity for Option B.** One file per entity is simple and matches the
  current sharding, but 100 000 small files is unkind to some filesystems and to
  backup tools. One file per parent group (a layer, a recording) matches the
  locality grouping `map_async` now uses and would be far friendlier, at the cost
  of write coordination between workers processing the same group.
- **Compression policy.** Off by default is the safe choice, since imaging traces
  compress poorly and CPU is the bottleneck; worth measuring on real data.
- **Non-array objects.** ASDF handles numpy and plain YAML-able structures well.
  Attributes that are currently arbitrary pickled Python objects (the
  `bs_cluster_full_indices` lists, for instance) need either a schema or a
  documented fallback.
- **Dependency weight.** `asdf` pulls in `jmespath`, `semantic-version` and
  `attrs`. Reasonable, but it should be an optional extra rather than a hard
  dependency.

## What I would need from you

- Whether archival and citability (Option C) or the blob layer (Option B) is the
  more pressing motivation, since that decides the order.
- Whether ~100 000 files is acceptable for Option B, or whether it should be one
  file per layer or recording.
- Whether anything currently reads the `ext/` tree directly, which would
  constrain how far the layout can change.
