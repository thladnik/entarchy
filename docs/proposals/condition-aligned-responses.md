# Proposal: responses aligned by stimulus condition, across recordings

Status: **proposed**. Nothing implemented. Written 2026-11-19 against entarchy
`c453e43` and vxentarchy `5cba591`.

Companion to [dense-matrices.md](dense-matrices.md), which this does not replace:
that proposal stores a pairwise result within one recording cheaply. This one is
about the question it leaves open — getting the same result for *a subset of ROIs
across all recordings*, with the stimulus phases lined up.

## The problem

An experiment is some animals, each with some recordings. Every recording has its
own stimulation phases, and the phases are the same protocol every time, more or
less: mostly the same number, drawn from the same sets of parameters. The job has
two halves:

1. **Per recording**, compute for each ROI its response to each of that
   recording's phases — the `mean_dff` of the dense-matrices proposal.
2. **Across recordings**, take any subset of ROIs — `[Animal]strain == "wt" AND
   has_receptive_field == True`, say — and get their responses to *the same
   stimulation phases*, as one table.

The first half is a loop. The second half is the one entarchy has no answer for,
and the reason is in the words "the same phases". Phases are entities, one set per
recording. A `Phase` in `rec_01` and a `Phase` in `rec_02` that showed the same
grating at the same azimuth are two entities with two uuids, and nothing in the
store says they are the same thing. Every tool entarchy has keys pairwise results
by uuid:

| today | why it does not do half two |
|---|---|
| `mean_response` links, `Phase -> Roi` | 1 KB per pair, 16.6 GB for the CMN set; and `matrix_from_links(phases, rois)` is still keyed by phase uuid, so phases from different recordings are different columns. Aligning them is a `groupby` on phase attributes afterwards, written by every caller. |
| `roi['mean_dff'] = array` | Positional against that recording's phase order. Comparable across recordings only while every protocol ran identically, and silently wrong the day one did not. |
| dense matrices (proposal) | One `Phase × Roi` block per recording, axes by uuid. Dense within a block, but the blocks share no axis, so the whole is 2.3% meaningful and there is nothing to stitch them on. |

So the missing piece is not storage. It is **a shared axis**: a key that says which
phases, across recordings, are the same condition.

## The move: a condition axis

Replace the per-recording `Phase` axis with a **condition** axis that every
recording shares. A condition is one combination of the stimulus parameters that
define the protocol — stimulus kind, azimuth, direction, contrast, whichever
parameters the experiment varied. Phases map onto conditions many-to-one: every
recording's `phase_17` and `phase_42` might both be `grating/azimuth=45`, and so
might `rec_02`'s `phase_16`.

This changes the shape of the whole result. Over the CMN set, `Phase × Roi` across
recordings is 726 M cells of which 17 M mean anything. `Condition × Roi` is on the
order of 300 × 42,521 = **12.8 M cells, every one meaningful, 51 MB as float32.**
The sparsity was never in the data; it was in using a per-recording axis for a
cross-recording question.

### Two ways to represent conditions

**(a) A condition key, derived.** A string built from the phase's parameters by a
function the schema defines once:

```python
class VxEntarchy(entarchy.Entarchy):
    condition_parameters = ('display/stimulus', 'display/azimuth', 'display/direction')

class Phase(entarchy.Entity):
    @property
    def condition(self) -> str:
        return condition_key(self, self.entarchy.condition_parameters)
        # -> 'display/stimulus=moving_grating|display/azimuth=45.0|display/direction=1'
```

Nothing is stored; no new type; the axis labels are strings. Grouping a
recording's phases by condition is `recording.get(Phase).dataframe_of([...])` and
a `groupby`. The weak point is "more or less the same": a float that is `45.0` in
one recording and `45.00001` in another is two conditions, and the fix — rounding,
a tolerance, a canonical parameter list — has to live in `condition_key` and
nowhere else. An attribute present in some recordings and absent in others is the
same problem. Both are protocol facts and belong in the schema, which is where (a)
puts them.

**(b) A `Condition` entity type.** A second root type, one entity per distinct
parameter set in the whole store, with each `Phase` linked to its condition:

```python
ent.add_child_entity_type(Condition)
ent.define_link_type('presents', Phase, Condition)
# at ingest, per phase:
ent.link(phase, ent.condition_for(phase), 'presents')
```

Now the axis is a collection like any other: `ent.get(Condition, 'stimulus ==
"moving_grating" AND azimuth > 0')` selects columns the way `ent.get(Roi, ...)`
selects rows, condition labels are uuids like every other axis, and a condition
has somewhere to hold condition-level results (a population tuning curve, a
preferred-azimuth histogram). The cost is the one noted in the experiment-as-link
discussion: the filter language does not traverse links, so "the phases that
present condition X" is a link query, not an ancestor filter, and the `Condition`
set has to be curated at ingest with the same tolerance rules (a) needs anyway.

**Recommendation: (a) first, with the key function and the parameter list in the
schema; (b) when a condition-level result needs a home.** (a) is a day's work and
makes the axis exist. (b) builds on it — the entity's id *is* the key — so nothing
written under (a) is thrown away.

**Not recommended: aligning by phase index.** If every protocol ran identically,
`phase.index` is the condition and the problem vanishes. It is also the assumption
that is true until a protocol is shuffled, shortened or extended, and the result is
a table that is wrong without being empty. It can be the key function for an
experiment where it holds; it should not be the mechanism.

## Storing the result, two ways

With a shared axis, each recording's result is a `Condition × Roi` matrix whose
rows are the same labels everywhere. Where it is written decides what reading a
subset of ROIs costs.

### (i) Per recording, on the analysis: blocks that concatenate

This is the dense-matrices proposal with the row axis labelled by condition key
rather than uuid (its open question "axes that are not entities", answered):

```python
with ent.current_analysis as analysis:
    for recording in ent.get(Recording):
        phases = recording.get(Phase)
        rois = recording.get(Roi)
        m = mean_dff_by_condition(phases, rois)          # DataFrame: condition × roi uuid
        analysis.set_matrix(f'mean_dff/{recording.uuid}', m.values,
                            rows=m.index, columns=rois)
```

Reading a subset across recordings needs no index of ROI to block, because a
collection can say which recording each member is in:

```python
rois = ent.get(Roi, '[Animal]strain == "wt" AND has_receptive_field == True')

table = rois.matrix_with(ent.conditions, 'mean_dff')
# Condition × Roi over all recordings: find the distinct [Recording]uuid of the
#  members, open those blocks only, take the members' columns, concat on the
#  shared row axis; conditions a recording lacked are NaN in its columns.
```

Thirty-eight blobs of ~1.7 MB, read whole, columns selected in memory. Blocks stay
independently readable. Writes are 38 big blobs, which suits a result computed per
recording in one go.

### (ii) Per ROI, as a vector against an axis stored once

Each ROI carries its own row: `roi['mean_dff']` is a float32 vector, and the row
labels are stored **once**, on the analysis, as an append-only list:

```python
analysis['mean_dff/axis'] = [...]                 # the condition keys, in order
rois.update(pd.DataFrame({'mean_dff': rows}, index=roi_uuids))   # one statement
```

Reading is what collections already do, plus a stack:

```python
table = rois.table_of('mean_dff')
# dataframe_of('mean_dff') gives one vector per ROI; table_of stacks them into
#  Roi × Condition, with the columns named from the axis on the analysis.
```

The subset is served entirely by the SQL filter that produced `rois`; nothing has
to find blocks. The cost is the convention: vectors are positional against the
stored axis, so the axis may only grow, a vector shorter than the axis is padded
with NaN on read, and a recording that lacks a condition writes NaN in that slot.
Labels are not duplicated per ROI, which is what makes it cheap: ~1.2 KB per ROI,
50 MB over the set. It is also a worse fit for a result that is naturally per
recording and pairwise, such as a per-trial response, where the block form keeps
the pairing explicit.

### Choosing

| | (i) blocks on the analysis | (ii) vectors on the ROI |
|---|---|---|
| write | 38 blobs, one per recording | 42,521 small blobs, one `update` per recording |
| read a ROI subset | open the blocks of its recordings, select columns | read the members' vectors, stack |
| what has to exist | `set_matrix`/`matrix_of` with a non-entity row axis; block stitching in `matrix_with` | an axis convention; `Collection.table_of` |
| ragged conditions | NaN rows within a block | append-only axis, NaN padding |
| the result belongs to | the analysis | the ROI, under the analysis's name |
| fits | pairwise results, per-trial detail, anything read per recording | per-ROI results read by ROI subset |

**Recommendation: (ii) for `mean_dff` and anything else that is a per-ROI vector
over conditions.** It answers the question asked — a subset of ROIs, their
responses, one table — with the machinery that already selects subsets, and the
new code is a stacking helper and an axis convention. (i) stays the right form for
genuinely pairwise data within a recording, and the two should share the
condition-key function so a result can move between them.

Neither form is filterable inside the vector: "ROIs whose response to condition X
exceeds 0.5" is not a `where`. Where that question matters, the answer is to also
write the few scalars it needs as plain attributes — `grating/peak_dff`,
`preferred_azimuth` — which are filterable, sortable and cheap. The vector is the
record; the scalars are the index into it.

## What the per-recording half looks like

The loop itself is ordinary, and the pieces exist: `recording.get(Phase)` and
`recording.get(Roi)` give the axes, the `phase_frames` links give each phase's
frame window in the imaging source, `map_async` over recordings parallelises it.

```python
def mean_dff_by_condition(recording: Recording) -> pd.DataFrame:
    phases = recording.get(Phase)
    rois = recording.get(Roi)
    dff = np.stack(rois.dataframe_of(['dff'])['dff'])            # roi × frame
    key = phases.dataframe_of(['uuid']).index.map(lambda u: ent.get_entity_by_uuid(u).condition)
    per_phase = pd.DataFrame({p.uuid: dff[:, s:e + 1].mean(axis=1)
                              for p, (s, e) in zip(phases, phases.frame_windows())},
                             index=rois.uuids)
    return per_phase.T.groupby(key).mean()                       # condition × roi

with ent.current_analysis as analysis:
    for recording in ent.get(Recording):
        m = mean_dff_by_condition(recording)
        analysis.extend_axis('mean_dff', m.index)                 # append-only
        recording.get(Roi).update(analysis.rows_for('mean_dff', m))
```

`frame_windows`, `extend_axis` and `rows_for` are sketches of the small helpers
the loop wants, not part of the proposal's surface; the shape of the loop is the
point. Repeated conditions within a recording — the same grating shown three times
— average in the `groupby`, which is usually what `mean_dff` means; a per-trial
result would keep them apart and is the case for form (i).

## What is entarchy's and what is vxentarchy's

**entarchy** gets the generic mechanism:

- `Collection.table_of(name)` — stack a vector attribute into a wide frame, columns
  named from an axis stored on the current analysis; NaN-pad short vectors.
- The axis convention: `analysis[f'{name}/axis']`, append-only, with
  `extend_axis` refusing to reorder or drop.
- For form (i): `set_matrix`/`matrix_of` accepting a label list as an axis in place
  of a collection, and `matrix_with` stitching blocks by the members' recording.

**vxentarchy** gets the protocol knowledge:

- `condition_parameters` on the schema and `Phase.condition`, with the rounding
  and canonicalisation that "more or less the same" needs.
- `mean_dff_by_condition` and the loop, as an analysis step next to the existing
  ones.
- Later, `Condition` as a root type and the `presents` link, if condition-level
  results want a home.

## Open questions

- **Float parameters.** Azimuth `45.0` versus `44.99999` is the same condition;
  `45` versus `50` is not. Rounding to a declared precision per parameter is the
  obvious rule. Whether the precision belongs in `condition_parameters` or in the
  stimulus definitions vxpy already has is a vxentarchy question.
- **Order of the axis.** Append-only means the axis is in order of first
  appearance, which is arbitrary. A sort key on the condition parameters for
  presentation is cheap; reordering storage is not, and should not happen.
- **One axis per name or one per analysis?** `mean_dff` and `peak_time` over the
  same conditions could share an axis. Per name is simpler and costs a few
  kilobytes; sharing invites a vector to be read against the wrong axis.
- **Trials.** A condition shown three times in a recording is three phases. The
  mean is what `mean_dff` wants; variance across trials, or per-trial responses,
  need the phases kept apart, which is form (i) with a `Phase` axis within the
  block and a condition key beside it. Worth deciding before the first table is
  written, because the two are not the same stored object.
- **Promoting conditions to entities.** When (b) happens, the key becomes the
  entity id and `presents` links are created at ingest. Results written under (a)
  are already labelled by that key, so they do not move. What does change is the
  `Condition` set itself having to be curated: a typo in a parameter would mint a
  condition entity rather than a stray column.
