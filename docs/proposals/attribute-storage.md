# Attribute storage: EAV against JSON columns

entarchy stores every attribute as a row in `attributes`, with seven typed value
columns and a `data_type` saying which one holds the value. That is an EAV
(entity–attribute–value) model, and EAV has a reputation for being slow. This
document asks whether JSON columns would be better, answers it with
measurements, and records the three fixes that came out of the exercise.

**Conclusion: keep EAV.** The document form wins on bulk load, storage and the
DataFrame pivot, but its filter advantage disappears once entities get wide or
the per-attribute metadata comes along, and it structurally cannot answer the
ancestor traversals entarchy's query language is built on. The three fixes below
take the available wins without changing the model.

## What was compared

Three schemas holding identical data:

| | shape |
|---|---|
| **A** `eav` | entarchy as it is: one row per attribute, seven typed value columns |
| **B** `json_value` | one row per attribute still, a single JSON `value` column instead of seven |
| **C** `json_doc` | one row per entity, every scalar in one JSON document; blobs in a side table |

Queries use entarchy's own shapes: the `uuid IN (subquery)` attribute filter from
`_generate_attribute_filters`, and the `MAX(CASE …) GROUP BY` pivot from
`get_collection_attributes`. Row counts were checked equal across schemas for
every query before any timing was compared.

Scripts are not in the repository; they are reproducible from this document.
Measurements are warm-cache medians on one machine — the ratios are the point,
not the absolute numbers.

## Results

27 000 entities × 13 scalar attributes, no blobs (identical in every design), in
milliseconds.

| | EAV | EAV lean | json_doc | jsonb_doc | MySQL EAV | MySQL doc |
|---|---|---|---|---|---|---|
| filter, 1 row | 73.6 | 73.9 | 54.1 | 23.9 | 258.4 | 71.7 |
| filter, ~53% of rows | 142.3 | 109.9 | 76.7 | 35.6 | 435.4 | 222.2 |
| **filter on parent** | **31.6** | **26.1** | 93.4 | 59.8 | **295.8** | 391.7 |
| pivot 5 attributes | 1000.9 | 1004.6 | **122.8** | 103.6 | — | — |
| pivot 13 attributes | 1252.4 | 1279.8 | **236.4** | 222.1 | 2453.7 | **1039.9** |
| filter 1 row, indexed | 8.3 | — | **0.1** | — | **0.5** | 0.7 |
| bulk load | 6325 | 4619 | **570** | 644 | 31735 | **2979** |
| size (MB) | 95.8 | 74.1 | **16.1** | 14.7 | 182.0 | **25.6** |

`EAV lean` is EAV without the duplicate index (fix 1). `jsonb_doc` needs SQLite
3.45+, so Python 3.12+; entarchy supports 3.10, which rules it out as a baseline.

**Option B is dead.** Row-per-attribute with a JSON value was slower than EAV on
every operation, needed a bigger index, and gave up type fidelity for nothing.
It is not discussed further.

### The advantage inverts with document width

Same data, 73 attributes per entity instead of 13:

| | EAV | json_doc |
|---|---|---|
| filter, 1 row | **209** | 332 |
| filter, ~53% | **255** | 365 |
| filter on parent | **24** | 534 |
| pivot 13 | 5465 | **627** |

EAV cost scales with the *rows touched*; the document scales with *width ×
entities*. Note that the parent traversal is **24 ms at either width** — the
index on `attributes.name` isolates the 50 rows named `depth` no matter how much
else exists. A document has no equivalent and must parse every entity row.

entarchy's query language leans on this constantly: `../depth > 0`,
`[Recording]imaging_rate > 8.0`, `@Roi.has_receptive_field == True`. Every one of
them is a "which entities have this attribute" lookup.

### The document's win is mostly an artifact of dropping metadata

`mutable` (immutability, enforced in `set_entity_attributes`), `analysis_uuid`
(which analysis wrote the value), `created`, `modified` and `data_size` are
per-attribute columns today. A flat `{name: value}` document holds none of them.
Nesting them as `{name: {v: …, m: …, a: …}}`:

| | size | document | filter | pivot |
|---|---|---|---|---|
| flat `{name: value}` | 14.5 MB | 406 B | 72.5 ms | 265.4 ms |
| with per-attribute metadata | 107.0 MB | 2057 B | 337.9 ms | 689.4 ms |
| EAV, same metadata | 80.2 MB | — | 126.1 ms | — |

Carrying what entarchy already carries, the document is **larger and slower on
filters than EAV**. Only the pivot still favours it.

### Type fidelity

| value | through a JSON column |
|---|---|
| `float('nan')`, `±inf` | `json.dumps` emits `NaN`/`Infinity`; SQLite rejects it as malformed JSON |
| `date`, `datetime`, `bytes` | `TypeError` — not serializable |
| `bool` | comes back from `json_extract` as `1` |
| int vs float | preserved (`json_type` reports `integer` / `real`) |

entarchy stores NaN and Inf today, with the sign of infinity kept in
`value_int` and flagged by `float_is_nan` / `float_is_inf`. A JSON column would
need all of that out of band anyway.

### Concurrent partial writes

Two workers setting different attributes of the same entity — what `map_async`
does when several analyses write results for the same ROI:

```
EAV (row per attribute)    -> {'result_0': 0.0, 'result_1': 1.0}   both kept
JSON document per entity   -> {'result_1': 1.0}                    one write lost
```

`json_set()` in SQL avoids the lost update where the new value can be expressed
in SQL, but it still rewrites the whole document.

## The three fixes

None of them changes what a query returns.

### 1. Drop the duplicate index

`attributes` had a primary key on `(entity_uuid, name)` *and*
`ix_unique_name_per_entity_uuid` on the same two columns in the same order.
Every dialect already backs a primary key with a unique index — SQLite as
`sqlite_autoindex_attributes_1`, InnoDB as the clustered index — so the second
one duplicated it exactly. `ANALYZE` reported identical statistics for both:

```
attributes  ix_unique_name_per_entity_uuid  554652 11 1
attributes  sqlite_autoindex_attributes_1   554652 11 1
```

On the 27 000-ROI entarchy it occupied **44.1 MB** (3.9% of a 1.12 GB file, 65
bytes per attribute row) plus its share of every insert. In the scalar-only
benchmark, dropping it took the file from 95.8 to 74.1 MB and bulk load from
6325 to 4619 ms.

The single-column index on `name` is a different matter and is kept: it is what
makes the ancestor traversals above cheap.

### 2. Restrict the pivot to the attributes asked for

`get_collection_attributes` builds a `CASE` per requested name but joined every
attribute row of every entity, letting each `CASE` discard what it did not want.
Restricting the join to the requested names, on 27 000 entities holding 73
attributes each:

```
want  5 of 73:  4452 -> 1170 ms   3.81x
want 13 of 73:  5648 -> 2139 ms   2.64x
want  5 of 13:   995 ->  763 ms   1.30x
want 13 of 13:  1342 -> 1571 ms   0.85x   <- the cost
```

The loss is confined to asking for every attribute a narrow entity has, where
the restriction can eliminate nothing. Entities accumulate attributes as
analyses add them, so the ratio moves the right way with the age of an entarchy.

**The restriction has to go in the join condition of an OUTER join.** Attributes
are per entity, not per type, so an entity may have none of the requested names;
as an inner join with the names in `WHERE`, such an entity produces no rows and
drops silently out of the result, where before it appeared with NaN. The outer
join costs nothing measurable over the inner one.

### 3. Keep SQLite's query planner statistics

Without a `sqlite_stat1` table SQLite guesses how selective an index is, and for
entarchy's filters it guessed badly — driving the query from the entity type,
reading every entity of that type, rather than from the attribute the filter
names. With a composite index available to choose:

```
before ANALYZE  20.18 ms  SEARCH entities USING INDEX ix_entities_type
after  ANALYZE   0.40 ms  SEARCH entities USING INDEX sqlite_autoindex_entities_1
```

The SQLite backend now sets `PRAGMA analysis_limit=400` on each connection and
runs `PRAGMA optimize` when it closes, which analyses only what the session
touched and what has changed enough to matter. Archives get one `ANALYZE` at
export, since their index is read-only afterwards. MySQL is unaffected: InnoDB
maintains its own statistics.

Existing databases have none at all, and get them from the tool below.

### Applying them to an existing entarchy

```
python -m entarchy.tools.optimize_storage <entarchy or URL>          # dry run
python -m entarchy.tools.optimize_storage <entarchy or URL> --apply
```

Safe to repeat. On SQLite the space the index held goes on the free list; run
`VACUUM` to return it to the filesystem.

## What the fixes are actually worth

On the 27 000-ROI test entarchy, through the API rather than in the benchmark:

| | as shipped | optimised | |
|---|---|---|---|
| filter `iscell == True` | 165.2 ms | 143.1 ms | 1.15x |
| filter `s2p/npix > 200` | 8.5 ms | 8.5 ms | 1.00x |
| filter `../depth > 0` | 20.6 ms | 21.0 ms | 1.00x |
| pivot query, 3 of 23 attributes | 1329.7 ms | 1035.3 ms | 1.28x |
| `dataframe_of`, 3 of 23, end to end | 2325.0 ms | 2139.3 ms | 1.09x |
| file size | 1146 MB | 1102 MB after VACUUM | 44.1 MB |

This is honest but underwhelming, and the reason is worth recording: the large
ratios in the benchmark need either wide entities or an index the planner can
choose between, and this entarchy has neither. The fixes are cheap and the
upside is real, but they are not a transformation.

## What is left

**`get_collection_attributes` runs two queries, and the other one dominates.**
Before the pivot it runs a `SELECT DISTINCT name, data_type` restricted to the
collection, purely to learn which value column each attribute lives in. On the
test entarchy that is **944 ms of the 2325 ms** a `dataframe_of` takes — more
than the pivot it precedes. Without the collection join it answers the same
thing in 49 ms, a 19x difference, but that changes semantics: the collection
restriction is what makes the "attribute has multiple data types" warning
meaningful. Worth doing, needs a decision about that warning first.

**Blob attributes are 27% of the rows** on the test entarchy and each holds a
pickled `Serializer` in `value_blob`, inline in the database unless the payload
exceeds `max_blob_size` (10 MB). This is orthogonal to EAV-versus-JSON — no
design here changes it — but it dominates the file size and is the subject of
its own open question.

**A long string is stored in `value_str`, which is `String(500)`.** SQLite
ignores declared lengths, so it round-trips; MySQL raises `DataError (1406)
Data too long`. Nothing routes an oversized string to the blob path, so the same
write succeeds or fails depending on the backend.
