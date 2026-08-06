# Proposal: dynamic link entities

Status: draft for discussion
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
  link_type       VARCHAR  PRIMARY KEY
  linker_type_pk  FK entity_types   NULL = any
  linked_type_pk  FK entity_types   NULL = any
  symmetric       BOOL     DEFAULT false
  description     VARCHAR  NULL
  created

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
declaration in neither order. This also rescues `@` as sugar, since `roi @ phase`
and `phase @ roi` both land correctly.

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
ent.define_link_type('correlated', Roi, Roi, symmetric=True,
                     description='Pearson r of dF/F, retained where p < 0.01')

for layer in ent.get(Layer):
    rois = layer.rois
    traces = np.stack([roi['dff'] for roi in rois])
    uuids = [roi.uuid for roi in rois]

    r = np.corrcoef(traces)
    i, j = np.triu_indices(len(rois), k=1)
    keep = np.abs(r[i, j]) > 0.6

    ent.link_from_frame(pd.DataFrame({
        'linker_uuid': [uuids[a] for a in i[keep]],
        'linked_uuid': [uuids[b] for b in j[keep]],
        'r': r[i, j][keep],
    }), 'correlated')
```

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
ent.define_link_type('preferred_stimulus', Roi, Phase)

for roi in ent.get(Roi, 'has_receptive_field == True'):
    responses = roi.links('mean_response').dataframe_of(['mean_dff'])
    best = responses['mean_dff'].idxmax()
    ent.link(roi, ent.get_entity_by_uuid(best), 'preferred_stimulus')
```

400 ROIs gives 400 links rather than 120,000 — the case links are ideal for.

### 10. Links between links

Because the carrier is an entity and endpoints are entity uuids, this needs no
extra machinery:

```python
ent.define_link_type('adaptation', LinkEntity, LinkEntity, directed=True,
                     description='response to a repeated stimulus vs. its first presentation')

ent.link(response_repeat_2, response_repeat_1, 'adaptation')['ratio'] = 0.68
```

Whether this is a good idea in practice is an open question, but it falls out of
the design rather than being designed in, and it is the natural home for
"how did this pairwise quantity change between conditions".

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
| `ent.link`, `Entity.links`, `__matmul__` | small | |
| `link_from_frame` and bulk helpers | medium | blocked on the two bugs below |
| `LinkCollection` + `_build_query_from_collection` branch | medium | join `links` before the existing filters |
| `@Type.attr` / `@linker.` / `@linked.` syntax | medium | parser plus one branch in `_generate_attribute_filters`, structurally the same as the existing `[Type]attr` case |
| `EXIST_LINK(kind)` filter | small | |
| archive export of `links` and `link_types` | small | `_LINK_COLUMNS` already exists |
| tests | large | |

## Prerequisites

Two existing defects block bulk link creation. Both were found while measuring
for this proposal and are worth fixing regardless.

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
path achieves ~0.4 ms per value. At that rate 120,000 links take ~21 minutes and
2.4 million take ~7 hours, which makes example 1 impractical at full scale. A core
`insert().values([...])` in chunks should bring entity creation into the same
range as the attribute path; a rough target is 3–4 minutes for 120,000 links,
though that is an estimate rather than a measurement.

This second one is not link-specific — it is the same cost paid ingesting 100,000
ROIs today.

## Open questions

- **Should `parent = linker` or nearest common ancestor?** Linker is O(1) and
  gives good archive/`map_async` locality when links are created per-linker.
  Nearest common ancestor (the Recording, for phase → roi) is more principled and
  groups better when links fan across many linkers, but costs a lookup per link.
  Linker is the current recommendation; worth revisiting if the first real use
  fans the other way.
- **Deleting an endpoint.** Links must not outlive their endpoints. Cascade
  delete, or refuse to delete an entity that still has links? Refusing is safer
  and matches the fact that entarchy has no general delete for entities today.
- **Should `LinkEntity` endpoints be materialised eagerly?** `link.linker` and
  `link.linked` as properties are convenient but each is an entity load; a
  collection-level prefetch will be wanted for anything iterating many links.
- **Is `link → link` (example 10) worth supporting deliberately**, or should the
  registry refuse `LinkEntity` endpoints until there is a concrete use?

## What I would need from you

- Which of the examples above are real for your work in the next few months, and
  which are speculative. That decides whether to build the full query syntax up
  front or start with kind plus endpoint filters.
- Whether cross-recording and cross-animal links (examples 4 and 5) need to work
  across *entarchies*, or only within one. Across would be a substantially larger
  design.
- Whether you want `@` sugar at all, or prefer `ent.link(...)` to be the only way
  to create one. Creating a database row from an operator is convenient but
  surprising, and `@` collides with matrix multiplication if entities ever become
  array-like.
