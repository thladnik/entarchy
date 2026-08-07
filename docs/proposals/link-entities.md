# Proposal: dynamic link entities

Status: **schema, registry and write API implemented**; `LinkCollection` and the
query syntax are still proposals. Both prerequisites listed below are done.

Bulk link creation measures **0.758 ms per link** (12,000 links with two
attributes each, SQLite), so example 1's 120,000 links take about 1.5 minutes and
1,467 bytes each. That is against 10.46 ms per entity when this started: the
transaction batching took it to ~3 ms, and `link_from_frame` inserting through
the core rather than the ORM took it the rest of the way.
Measured against: entarchy at `559998a`, SQLite 3.40.1, MySQL 8.0.46, Windows 11 / Python 3.10

## Summary

An entarchy is a tree. Every entity has exactly one parent, and that single
hierarchy — Animal > Recording > Layer > Roi — carries all structural meaning.
Most interesting relationships in the data are not that tree: a ROI's response to
a stimulation phase, the same cell recorded on two days, two ROIs whose activity
correlates, a functional cluster spanning three imaging planes. None of these can
be expressed as parenthood, and all of them have data attached to the
*relationship* rather than to either end.

This proposes **links**: a relationship between two entities that is itself an
entity, and therefore carries arbitrary attributes through the machinery that
already exists.

The central design decision is that a link's **kind is data, not a class**.
`ent.link(phase, roi, 'mean_response')` should feel exactly like
`roi['mean_response'] = …` feels — invented on the spot, no declaration, no code
change. Kinds are constrained by their endpoint types, but those constraints are
recorded in the database rather than in Python.

Links are the right tool for *sparse* relationships. For dense pairwise results —
an all-to-all correlation matrix — they are the wrong tool by two orders of
magnitude, and this document says where the line falls and what to do instead.

## What the tree cannot express

Six relationships from the existing analysis, none of which fit the hierarchy:

| relationship | endpoints | why not the tree |
|---|---|---|
| response of a ROI during a stimulation phase | Phase → Roi | Phase and Roi are siblings under Recording; neither contains the other |
| the same cell imaged on a later day | Roi → Roi | different Recordings, different Layers |
| significantly correlated ROI pairs | Roi ↔ Roi | many-to-many within and across Layers |
| directed connectivity with lag | Roi → Roi | two different values per pair, one per direction |
| functional cluster membership | Roi → cluster | a ROI already has a parent, and may belong to several clusters |
| a plane's cells continuing into the next plane | Layer → Layer | ordering, not containment |

The pattern is the same each time: the tree gives you exactly one hierarchy, and
these are the other ones. The data belongs to the pair, so there is no entity it
can currently hang on without being duplicated or encoded into an attribute name.

## Design: the kind is data

entarchy is already split into a closed part and an open part. Entity types are
closed — Python classes, validated against `entarchy.yaml` on every open:

```python
if self._config['hierarchy'] != self._hierarchy:
    raise RuntimeError('Entity type hierarchy in configuration does not match…')
```

Attributes are open — `roi['dff'] = …` invents `dff` on the spot.

A link kind belongs on the open side. If a kind were an entity type, inventing
one would mutate the hierarchy and lock out anyone opening the entarchy without
that class. Whereas `LinkEntity` is *already* in the hierarchy dict, so if every
link shares that one class and the kind is a column, **the type machinery needs
no changes at all**. Dynamic kinds are the cheap option, not the expensive one.

The consequence worth stating plainly: an exported archive of your data is
readable, queryable and complete without any of your link definitions existing
anywhere. That is the same property the ASDF archive format was built to protect.

## Schema

```
link_types
  link_type         VARCHAR  PRIMARY KEY
  linker_type_pk    FK entity_types  NULL     -- endpoint is an ordinary entity
  linker_link_type  FK link_types    NULL     -- endpoint is a link of this kind
  linked_type_pk    FK entity_types  NULL
  linked_link_type  FK link_types    NULL
  symmetric         BOOL     DEFAULT false
  description       VARCHAR  NULL
  created
  -- per endpoint, at most one of (type_pk, link_type) is set;
  -- both null means a deliberate wildcard

links
  link_uuid       CHAR(36) PRIMARY KEY        -- also the carrier entity's uuid
  link_type       VARCHAR  FK link_types
  linker_uuid     CHAR(36) FK entities
  linked_uuid     CHAR(36) FK entities
  created, modified

  UNIQUE (link_type, linker_uuid, linked_uuid)
  INDEX  (link_type, linked_uuid)
  INDEX  (linker_uuid), (linked_uuid)
```

Creating a link writes two rows: an `entities` row (type `LinkEntity`,
`parent_uuid = linker`, `id = f'{link_type}@{linked_uuid}'`) and a `links` row.
Attributes then work unmodified, because they only need an `entities` row to
point at.

Four notes on those choices.

**`link_uuid` as primary key**, replacing the existing nullable `entity_uuid`.
Every link is an entity; there is no second class of link with nowhere to put
data. If relation-only links later prove common enough to matter, the refinement
is to create the `entities` row lazily on first attribute write, not to make the
column nullable.

**`parent = linker` is load-bearing, not cosmetic.** The archive exporter groups
block files by parent and `map_async` orders work by parent for worker locality.
Parentless links would collapse into a single group and lose both. It also keeps
`../` traversal meaningful from a link.

It beats the alternative of using the nearest common ancestor, for a reason that
only shows up at scale. In example 1 the linker is the Phase, so 300 phases ×
400 ROIs groups into 300 files of 400 links each — roughly 0.16 s to open one.
Nearest common ancestor would be the Recording, giving a single file with 120,000
tree entries, which at the measured 0.4 ms per entry is about 48 seconds to open
before any array can be read.

The consequence to remember: an archive holding many links has many block files,
so `ArchiveBackend(open_file_limit=...)` should be raised above the default 64
when a read pass sweeps all of them.

Links are children of their linker, so they never appear in an existing entity's
ancestor chain — a Roi's ancestors are still Layer, Recording, Animal. Existing
`[ParentType]attr` traversal is unaffected.

**`link_type` is part of the unique key**, which is what allows one phase/ROI pair
to carry both a `mean_response` and a `peak_latency` link. The current table's
composite primary key `(linker_uuid, linked_uuid)` forbids that outright.

**Nothing writes to `links` today**, so this is a redefinition rather than a
migration. `_LINK_COLUMNS` in `entarchy/tools/archive.py` needs updating
alongside, and `link_types` must be exported too — otherwise an archive would
carry links whose semantics were lost.

## Endpoint types and direction

### Constraints are recorded on first use

```python
# implicit: the first use defines the kind from the actual endpoints
ent.link(phase, roi, 'mean_response')          # registers Phase -> Roi

ent.link(animal_a, animal_b, 'mean_response')
# LinkTypeError: 'mean_response' connects Phase -> Roi, got Animal -> Animal

# explicit: when you want to be deliberate, or to add a description
ent.define_link_type('mean_response', Phase, Roi,
                     description='trial-averaged dF/F of a ROI during a phase')
```

Exact type match. The hierarchy is containment rather than subtyping, so there is
no inheritance to accommodate; `None` is an explicit wildcard for kinds that
deliberately accept anything.

**Endpoints that are themselves links are constrained by kind, not by type.**
Every link shares the entity type `LinkEntity`, so constraining by entity type
would permit an `adaptation` link between a `mean_response` and a `correlated` —
precisely the confusion the registry exists to prevent, at the one place the
mechanism cannot see. Hence the paired `*_link_type` columns:

```python
ent.define_link_type('adaptation', 'mean_response', 'mean_response', directed=True)
# constrains both endpoints to mean_response links, not merely to "some link"
```

The one hole in first-use registration is that if the *first* use is the mistake,
the registry learns the mistake. Mitigated by making the registry visible
(`ent.link_types()` prints it) and correctable (`redefine_link_type`, refusing
while links exist unless `delete_existing=True`). Pipelines that cannot tolerate
this declare up front.

Validation must cover the bulk path too, which costs one extra query: select
distinct `entity_type_pk` over the linker uuids and over the linked uuids and
compare against the registry — before inserting a million rows, not after.

### Direction is the default; symmetry is declared

Direction is not an extra feature bolted onto links. Once a kind declares its
endpoint types, `linker` and `linked` have distinct named roles by construction.

- **Endpoint types differ → directed, unambiguously.** Nothing to declare.
- **Endpoint types are the same → the kind must say** `symmetric=True` or
  `directed=True`. This is the only case where it cannot be inferred, so it is
  the only case where you are asked.

Symmetric kinds store canonically as `(min(uuid), max(uuid))` and are found from
either end.

Two consequences worth building in deliberately:

**Auto-orientation.** With `(Phase, Roi)` declared,
`ent.link(roi, phase, 'mean_response')` stores Phase → Roi rather than raising.
When the endpoint types differ, the intended orientation is unambiguous, so
strictness there would be pedantry. It errors only if the pair matches the
declaration in neither order.

**Query syntax follows the flag.** For directed kinds, `@linker.attr` and
`@linked.attr` are meaningful. For symmetric kinds they are not — canonical
ordering makes which-is-which an artifact of uuid sort order — so those forms
raise there and only `@Type.attr` / `@either.attr` are accepted. Refusing beats
returning something that depends on how the uuids happened to sort.

## Worked examples

### 1. Stimulus response — the motivating case

Directed, different endpoint types, created in bulk from computed values.

```python
ent.define_link_type('mean_response', Phase, Roi,
                     description='trial-averaged dF/F during the phase window')

for recording in ent.get(Recording):
    phases = recording.phases
    rois = recording.rois

    records = []
    for phase in phases:
        start, end = phase['ca_start_index'], phase['ca_end_index']
        for roi in rois:
            window = roi['dff'][start:end]
            records.append({'linker_uuid': phase.uuid,
                            'linked_uuid': roi.uuid,
                            'mean_dff': float(window.mean()),
                            'peak_dff': float(window.max()),
                            'n_frames': int(end - start)})

    ent.link_from_frame(pd.DataFrame(records), 'mean_response')
```

Then the queries that motivated the whole thing:

```python
# strong responses to a particular stimulus, in ROIs with a receptive field
ent.links('mean_response',
          '@Phase.index == 7 AND @Roi.has_receptive_field == True AND mean_dff > 0.3')

# every response of one ROI, ordered by stimulus
roi.links('mean_response').dataframe_of(['mean_dff', 'peak_dff'])

# responses from recordings above a given imaging rate
ent.links('mean_response', '@linker.[Recording]imaging_rate > 8.0')
```

### 2. Sparse thresholded correlation

Same endpoint types, so `symmetric` must be declared. Compute densely in memory,
persist only what survives thresholding.

```python
ent.define_link_type('correlated', Roi, Roi, symmetric=True, cardinality='sparse',
                     description='Pearson r of dF/F, retained where |r| > 0.6')

for layer in ent.get(Layer):
    rois = layer.rois
    r = np.corrcoef(np.stack([roi['dff'] for roi in rois]))

    ent.link_from_matrix(rois, rois, r, 'correlated',
                         where=lambda v: abs(v) > 0.6,
                         value_name='r', symmetric=True)
```

`link_from_matrix` takes the predicate as a required argument, so the
thresholding step cannot be omitted — see the guard rails section below for why
that matters more than it looks.

Found from either end, because the kind is symmetric:

```python
roi.links('correlated')                       # partners in either direction
ent.links('correlated', 'r > 0.8 AND @Roi.quality == "good"')
```

Note what dynamic kinds buy here: `'correlated_p01'` and `'correlated_p001'` are
two thresholds of the same analysis, and choosing to keep both is a decision made
at the REPL, not a schema change.

### 3. Directed connectivity — where symmetry would lose information

Same endpoint types, but explicitly directed, because both directions exist and
carry different values.

```python
ent.define_link_type('granger', Roi, Roi, directed=True,
                     description='Granger causality, linker -> linked')

# a -> b and b -> a are two separate links with two separate values
ent.link(roi_a, roi_b, 'granger')['f_statistic'] = 4.2
ent.link(roi_b, roi_a, 'granger')['f_statistic'] = 0.3
```

```python
# what does this ROI drive?
roi.links('granger', direction='out')
# what drives it?
roi.links('granger', direction='in')
```

The same shape covers lagged cross-correlation, where
`xcorr(A,B)(τ) = xcorr(B,A)(−τ)`. Storing that symmetrically with a signed lag is
possible, but then every reader must know the sign convention, and the first
person who forgets it silently flips a result. A directed kind makes the
convention structural.

### 4. The same cell across sessions

Endpoints in entirely different parts of the tree — different Layers, different
Recordings, possibly different days. This is the relationship the hierarchy is
least able to express.

```python
ent.define_link_type('same_cell', Roi, Roi, directed=True,
                     description='follow-up ROI (linker) matched to reference ROI (linked)')

ent.link(roi_day2, roi_day1, 'same_cell').update({
    'match_distance_um': 3.4,
    'match_confidence': 0.91,
    'method': 'ants_centroid',
})
```

```python
# longitudinal: every ROI matched to this reference cell
reference_roi.links('same_cell', direction='in')

# only confident matches, and only from a later recording
ent.links('same_cell',
          'match_confidence > 0.8 AND @linker.[Recording]id == "rec_02"')
```

Directed matters here for a reason unrelated to the values: it records *which end
is the reference*, which is information nothing else in the dataset holds.

### 5. Overlapping group membership

A ROI's parent is its Layer and always will be. Any other grouping — functional
clusters, response types, anatomical assignments — must be a link, and a ROI may
belong to several at once.

```python
# cluster nodes are ordinary entities; AnalysisEntity already exists for this
with ent:
    cluster = AnalysisEntity(ent, _id='cmn_cluster_03', _parent=ent.root)
    ent.add_new_entity(cluster)
    cluster['method'] = 'ward'
    cluster['centroid_eta'] = centroid

ent.define_link_type('cluster_member', Roi, AnalysisEntity)

ent.link_from_frame(pd.DataFrame({
    'linker_uuid': member_uuids,
    'linked_uuid': [cluster.uuid] * len(member_uuids),
    'silhouette': silhouette_scores,
}), 'cluster_member')
```

```python
cluster.links('cluster_member', direction='in')            # members
roi.links('cluster_member')                                # every cluster this ROI is in
ent.links('cluster_member', 'silhouette > 0.5 AND @Roi.has_receptive_field == True')
```

Many-to-many, overlapping, and revisable without touching the tree.

### 6. Receptive-field overlap

From the CMN analysis. Symmetric, sparse after thresholding, and the link carries
a derived quantity that belongs to neither ROI.

```python
ent.define_link_type('rf_overlap', Roi, Roi, symmetric=True)

ent.link(roi_a, roi_b, 'rf_overlap').update({
    'jaccard': 0.62,
    'shared_patch_indices': np.array([3, 7, 11]),      # arrays work like any attribute
    'centroid_distance_deg': 8.1,
})
```

The array attribute is the point: links get the full blob machinery, including
`ext/` spill and ASDF archiving, with no extra work.

### 7. Anatomical adjacency between planes

```python
ent.define_link_type('z_adjacent', Layer, Layer, directed=True,
                     description='linker is immediately above linked')

ent.link(plane0, plane1, 'z_adjacent')['dz_um'] = 15.0
```

Useful for volumetric analysis where a cell spans planes, and for propagating
registration between neighbouring planes.

### 8. Provenance and quality control

```python
ent.define_link_type('flagged_by', Recording, AnalysisEntity,
                     description='QC finding attached to a recording')

ent.link(recording, qc_run, 'flagged_by').update({
    'reason': 'motion artifact',
    'severity': 'high',
    'frame_range': np.array([1204, 1387]),
})

# exclude flagged recordings from an analysis
clean = ent.get(Recording, 'NOT(EXIST_LINK(flagged_by))')
```

### 9. Preferred stimulus — sparse, and the reverse of example 1

One link per ROI rather than one per pair. A different kind, so the opposite
orientation is not a conflict.

```python
ent.define_link_type('preferred_stimulus', Roi, Phase, cardinality='one_per_linker')

for roi in ent.get(Roi, 'has_receptive_field == True'):
    responses = roi.links('mean_response').dataframe_of(['mean_dff'])
    best = responses['mean_dff'].idxmax()
    ent.link(roi, ent.get_entity_by_uuid(best), 'preferred_stimulus')
```

400 ROIs gives 400 links rather than 120,000 — the case links are ideal for.

### 10. Links between links

Supported deliberately. The carrier is an entity and endpoints are
`entities.uuid` foreign keys, so this needs no schema change — *refusing* it
would cost extra code.

```python
ent.define_link_type('adaptation', 'mean_response', 'mean_response', directed=True,
                     description='response to a repeated stimulus vs. its first presentation')

ent.link(response_repeat_2, response_repeat_1, 'adaptation')['ratio'] = 0.68
```

The endpoints are constrained by *kind*, which is why the registry needs the
`*_link_type` columns; constraining by entity type would allow an `adaptation`
between a `mean_response` and a `correlated`.

Deferred until there is a concrete use: addressing a link endpoint's attributes
in a filter, e.g. `@mean_response.mean_dff > 0.3`. That is real parser work, and
storage plus constraints are useful without it.

Two consequences to accept knowingly:

- **Deletion becomes recursive.** Refusing to delete an entity that still has
  links means deleting a ROI requires clearing its responses first, which
  requires clearing any adaptation links above those. The refusal has to explain
  a chain of unbounded depth.
- **Paths get deep.** With `parent = linker`, a link-of-a-link sits two levels
  below the ROI, so `entity.path` grows accordingly. Cosmetic.

### 11. Analysis over links, free

`LinkCollection` subclasses `Collection`, so everything that works on entities
works on links the day it exists:

```python
responses = ent.links('mean_response', '@Roi.has_receptive_field == True')

responses.map_async(functions.fit_response_kinetics, _worker_num=6)
responses.dataframe_of(['mean_dff', 'tau_rise', 'tau_decay'])
responses.to_asdf('/archive/figure_4_responses')
```

That last line matters: a link collection exports and reopens like any other
collection, and the archive carries `link_types`, so the semantics travel too.

## Where links are the wrong tool

Measured cost of an entity with attributes:

| | per entity |
|---|---:|
| SQLite, 4 attributes | 1,812 B |
| MySQL, 3 attributes | 1,956–2,091 B |

Roughly 2 kB per link. Against that, one `float32` is 4 bytes.

| case | links | as entities | as a float32 matrix |
|---|---:|---:|---:|
| 400 ROIs × 300 phases, one recording | 120,000 | 0.23 GB | — |
| × 20 recordings | 2,400,000 | 4.7 GB | — |
| 400 ROIs pairwise, one layer | 79,800 | 0.16 GB | 0.0006 GB |
| 100k ROIs pairwise, whole dataset | 5.0 × 10⁹ | **8,400 GB** | **37 GB** |

The stimulus-response case is comfortable. All-to-all correlation is not: a 227×
storage penalty for wrapping four bytes in two kilobytes of entity, and at dataset
scale it is simply impossible.

**Dense pairwise results belong on the nearest common ancestor as a matrix:**

```python
layer['roi_correlation'] = r.astype(np.float32)         # (n, n)
layer['roi_correlation_index'] = np.array(uuids)        # row/column order
```

This is also the form the data is consumed in — clustering and plotting want a
matrix, not 80,000 row lookups.

The rule I would write down: **entity ↔ entity relationships are links;
collection ↔ collection relationships are matrices.** Example 2 shows the hybrid,
which is usually what you actually want: compute densely, threshold, persist the
survivors as links.

## Guard rails against accidental density

Nothing in the design so far stops someone omitting the threshold in example 2
and writing every pair. That needs fixing, because the failure mode is quiet.

The catastrophic case is the safe one: 5 × 10⁹ pairs needs roughly 500 GB just to
build the DataFrame, so it dies in pandas before reaching the database. The
damage happens in the middle, where the write *succeeds slowly*:

| pairwise over | links | storage | write time now | after the bulk insert fix |
|---|---:|---:|---:|---:|
| 400 ROIs, one layer | 79,800 | 0.16 GB | 13 min | ~2 min |
| 3,200 ROIs | 5.1 × 10⁶ | 10 GB | 14 h | ~2 h |
| 10,000 ROIs | 5.0 × 10⁷ | 100 GB | 6 days | ~21 h |

Note that fixing the bulk insert sharpens this rather than softening it. A
runaway write is currently so slow it would be noticed and killed; at 1.5 ms per
link it fills a disk overnight instead. The per-layer loop in example 2 hides it
further, since each iteration looks modest and only the total is ruinous.

Five measures, which catch different mistakes:

**Cardinality declared on the kind.** The registry already exists, so expected
shape belongs in it:

```python
ent.define_link_type('correlated', Roi, Roi, symmetric=True, cardinality='sparse')
ent.define_link_type('preferred_stimulus', Roi, Phase, cardinality='one_per_linker')
```

`one_per_linker` is exactly enforceable with a unique index on
`(link_type, linker_uuid)` — not a heuristic — and would catch the example 9 bug
of writing 300 preferred stimuli per ROI. `sparse` is the default and enables the
density check; `dense` switches the checks off, deliberately and on the record.

**Density, not just count.** 500,000 sparse links across a large dataset is
legitimate; 80,000 links that are 100% of the available pairs is a misuse. On a
bulk write, compare `len(df)` against `unique_linkers × unique_linked` and refuse
above ~0.5 for a sparse kind, naming the density and pointing at the matrix form.

**A count ceiling with explicit override**, following the existing
`backend.delete(confirm=True)` idiom:

```python
ent.link_from_frame(df, 'correlated')                     # raises above ~100k
ent.link_from_frame(df, 'correlated', confirm_count=79_800)
```

Requiring the actual number rather than `force=True` means the caller has looked
at it.

**Make thresholding structural.** The strongest measure, because it removes the
mistake rather than detecting it — hand over the matrix and a predicate instead
of a pre-built frame:

```python
ent.link_from_matrix(rois, rois, r, 'correlated',
                     where=lambda v: abs(v) > 0.6, symmetric=True)
```

The threshold is a required argument, so it cannot be forgotten, and the helper
reports how many pairs survived before writing anything. Example 2 should be
written this way.

**`dry_run=True`** on the bulk paths, reporting count, estimated storage and
estimated time without writing. The thing to reach for the first time a new
linking script runs.

Every one of these needs a single-argument override. Someone who genuinely wants
200,000 links should get them without editing configuration.

## What this reuses, and what has to be built

Reused unchanged: the attribute table and all typed columns, the `Serializer` and
blob spill, `ext/` storage, filter-expression parsing and the query builder's
subquery machinery, `Collection` slicing and `dataframe_of`, `map_async`, digest
mode, the ASDF exporter.

To be built:

| piece | size | notes |
|---|---|---|
| `link_types` + `links` schema | small | redefinition, no data migration |
| registry: define / validate / introspect | small | one query per bulk validation |
| `ent.link`, `Entity.links` | small | `Entity.__matmul__` stays unimplemented; it should raise a message pointing at `ent.link` rather than the current bare NotImplementedError |
| `link_from_frame` and bulk helpers | medium | blocked on the two bugs below |
| `link_from_matrix` with required predicate | small | removes the example 2 failure mode |
| cardinality: `one_per_linker` index, density and count checks, `dry_run` | small | see the guard rails section |
| `LinkCollection` + `_build_query_from_collection` branch | medium | join `links` before the existing filters |
| `@Type.attr` / `@linker.` / `@linked.` syntax | medium | parser plus one branch in `_generate_attribute_filters`, structurally the same as the existing `[Type]attr` case |
| `EXIST_LINK(kind)` filter | small | |
| archive export of `links` and `link_types` | small | `_LINK_COLUMNS` already exists |
| tests | large | |

## Prerequisites

Two existing defects blocked bulk link creation. Both were found while measuring
for this proposal and were worth fixing regardless. **Both are now fixed**; the
diagnosis of the second turned out to be wrong, and the measurements below record
what it actually was.

**`collection.update()` fails above ~2,340 entities on SQLite.**
`set_collection_attributes` builds a single INSERT with every row inline; SQLite
caps bound parameters at 32,766, and at ~14 columns per row that is ~2,340 rows.
MySQL is unaffected because PyMySQL interpolates literals rather than binding
parameters — the mirror image of the DATETIME divergence. Fix is chunking, ~10
lines plus a test at 5,000 rows.

**Entity creation does not scale with batch size.** `add_entities` uses ORM
`session.add_all()`, a unit-of-work insert rather than a bulk one:

| entities | entity rows | attributes (×3) | per entity |
|---:|---:|---:|---:|
| 5,000 | 42.7 s | 5.7 s | 9.68 ms |
| 20,000 | 185.4 s | 23.8 s | 10.46 ms |

Flat at ~10 ms per entity however large the batch, while the collection attribute
path achieves ~0.4 ms per value.

Profiling showed the diagnosis was wrong. `add_entities` was already 0.14 ms per
entity; the cost was in the per-entity attribute commit that follows it, since
`add_new_entity` writes `id` and `uuid`. Of that, `sqlite3.Connection.commit`
alone was 2.98 s of 5.35 s — a commit is an fsync, measured at 5.91 ms against
0.028 ms for the same insert inside a transaction. `Entarchy.commit()` now runs
as one batch, which brings entity creation to about 3 ms and makes a block
all-or-nothing rather than half-persisted on failure.

What remains at ~3 ms is SQLAlchemy ORM overhead, not the database: `sqlite3`
execute and executemany together account for 0.24 s of 7.7 s. Going below that
means bypassing the ORM, which the bulk link writer should do rather than the
general entity path — keeping the risky surgery in new code.

This second one was never link-specific: it is the same cost paid ingesting
100,000 ROIs today.

## Decisions

Settled, and reflected above.

- **`parent = linker`**, not nearest common ancestor. O(1), and it produces the
  block-file granularity the archive format wants (see the schema notes).
- **Deleting an entity that still has links is refused**, rather than cascading.
  Safer, and it matches the fact that entarchy has no general entity delete
  today. With link → link supported, the refusal walks a chain.
- **Link endpoints are prefetched at collection level.** `link.linker` and
  `link.linked` as per-link properties would be one entity load each; anything
  iterating a `LinkCollection` needs them resolved in bulk.
- **`link → link` is supported** (example 10), with endpoints constrained by kind
  rather than entity type. The registry columns go in now, while the table is
  being created; the filter syntax for addressing a link endpoint's attributes
  waits for a real use.
- **No `@` operator.** `ent.link(...)` is the only way to create a link. Creating
  a database row from an operator is surprising, and `@` would collide with
  matrix multiplication if entities ever gain array-like behaviour.
- **All eleven example kinds are plausible for real use**, so the query language
  is built up front rather than deferred — endpoint filters are what make links
  worth having, and every example above depends on them.

## Not in scope: links across entarchies

Tempting for cross-animal or cross-experiment work, but it fails in five ways,
in increasing order of severity:

1. **No referential integrity.** `links` has a foreign key into `entities` in one
   database. Across two, a dangling reference is undetectable until read, and the
   delete guard above is blind to the far side.
2. **No transactional consistency.** Two databases, no two-phase commit.
3. **Backend heterogeneity.** One entarchy on SQLite and one on MySQL cannot be
   joined at all.
4. **The query engine would silently half-work.** `@Roi.has_receptive_field ==
   True` compiles to a SQL subquery; if the endpoint lives elsewhere that is a
   cross-database join — impossible on SQLite, and on MySQL only when both
   schemas sit on one server with the right grants. A filter language that
   behaves differently depending on deployment is worse than one that refuses.
5. **It breaks archive self-containment.** A cross-entarchy link must name the
   other entarchy: by path, which breaks on any move or copy, or by uuid plus a
   discovery registry, which means archives need an external resolver. That
   directly undoes the property the ASDF archive format was built to guarantee.

The need behind the idea is real, and there are two better answers.

**Put the animals in one entarchy.** The hierarchy already roots at Animal, so
several animals in one entarchy is the intended design, and cross-animal links
are then ordinary links. For entarchies that are already separate, the thing to
build is a **merge tool** — uuid4 collisions are not a practical concern, and it
is adjacent to the existing `import_archive` code. That converts a distributed
reference problem into a one-time data operation.

**Otherwise store the correspondence as a plain attribute**, and be explicit
that it is a weak reference nothing enforces:

```python
roi['external_match'] = {'entarchy_uuid': '…', 'roi_uuid': '…', 'confidence': 0.9}
```

Honest about what it is, rather than dressed as a link and implying an integrity
guarantee that does not exist.

## Remaining questions

- **Merge tool: wanted?** Only if per-experiment entarchies already exist that
  need relating. If new work goes into one entarchy per project, this is moot.
- **Does refusing deletion need an escape hatch** — a `delete_links=True` that
  clears the chain — or is manual cleanup acceptable given how rarely entities
  are deleted?
