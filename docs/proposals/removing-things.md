# Proposal: removing attributes and entities

Status: **implemented** 2026-11-25, as written, with the open questions decided
as the text proposes. Deviations: a committed deletion reads back as any missing
attribute does, with `AttributeError`, while a deletion still queued raises
`KeyError`; `Entity.is_removed` says whether a handle is dead; a removal inside
`with ent:` is not deferred, and drops what the context had queued for the
removed entities. The MySQL backend's foreign keys are exercised in
`tests/test_mysql_integration.py`.

Written 2026-11-25 against entarchy `895e5d7`, with the vxpy entarchy at
`Z:/cluster/data/zfish-egomotion` (MySQL) as the case at hand.

## Summary

Entarchy can create entities, write attributes and link entities, and it can
delete the whole store. It cannot remove one attribute or one entity. Today that
cost an afternoon: an ingest failed at the frame timing step and left eight
recordings each with an `Imaging` entity that has three attributes and no
layers. The only way to take those out is a script that deletes rows from three
tables by hand, which is what `scripts/repair_texasred.py` in vxpy-entarchy
does. The link machinery has had its own removal (`remove_links_of_type`) since
links were built, because a link kind that was defined wrongly had to go. The
same need exists for everything else.

This proposes two operations, each at the level where things are added:

```python
del roi['cmn_rf/dff']                     # one attribute, one entity
del rois['cmn_rf/dff']                    # one attribute, a whole collection

recording.remove_child(imaging)           # an entity without children
animal.remove_child(recording, recursive=True)   # a subtree
ent.remove_child(experiment, recursive=True)     # forwards to the root, as add_child does
```

They follow the rules writes already follow: queued inside `with ent:`,
immediate outside; refused for immutable attributes outside digest mode; never
for `id` and `uuid`. Removing an entity takes its links with it, in both
directions, and refuses a subtree unless asked for one. Each prints what it is
about to remove before it does, and a `dry_run` returns that report instead.

## What a removal has to touch

More than rows. The table is what `add` builds up, read backwards.

| State | Where | On removing an attribute | On removing an entity |
|---|---|---|---|
| Attribute row | `attributes` | delete | delete all rows of the entity |
| Value in a file of its own | `ext/<uuid shards>/<name hash>.blob` | remove the file | remove every owned file |
| Media file | `media/<uuid shards>/<name hash>.<ext>` | remove the file | remove every owned file |
| Value in an archive block | `asdf:<file>#<key>` | row goes, block stays | same |
| Link rows | `links`, by `linker_uuid` or `linked_uuid` | - | delete, and their carrier entities |
| Carrier entities | `entities`, `LinkEntity` children of the linker | - | delete with the link rows |
| Child entities | `entities`, by `parent_uuid` | - | refuse, or delete deepest first |
| Entity row | `entities` | - | delete, last |
| Identity map | `Entarchy._entities` | - | pop; mark the object removed |
| Attribute cache | `Entity._attribute_cache` | pop the name | drop |
| Pending writes | `_entities_to_add`, `_entities_to_update`, `_links_to_add` | drop the name from the queue | dequeue; a never-committed entity needs no row deletes |
| Collection caches | `Collection._cache` | stale for that column | stale for that row |

Three of these deserve a word.

**Files are removed after the transaction commits, never inside it.** A
transaction that fails must not have deleted files; a file left behind after a
commit is a leak, not a corrupt store. That is the order `_discard_replaced_file`
already uses when a value is overwritten by a smaller one, and the removal
collects the owned paths with `_owned_file_of` before it deletes any row, then
removes them once the commit returned. A failed `os.remove` is logged, not
raised, for the same reason it is swallowed today.

**Archived values are not freed.** An ASDF archive packs many values into one
block file, and a block cannot be removed without rewriting the file. Removing
such an attribute deletes the row and leaves the block; the space comes back when
the archive is repacked. `describe` can report orphaned blocks. This is the
trade the archive made when it was built: a file written once by the archive
tooling and only read afterwards, in exchange for mmap-able reads.

**Foreign keys are enforced on MySQL and not on SQLite.** The schema declares
`parent_uuid`, `entity_uuid` and the three link columns as foreign keys. InnoDB
enforces them; the SQLite backend never issues `PRAGMA foreign_keys = ON`. So
the delete order is fixed by MySQL and merely good manners on SQLite:
attributes, then links, then entities deepest first. Nothing relies on cascades,
which neither schema declares, so a removal is explicit about every row it
takes - which is also what makes the report possible.

## Attributes

### API

```python
del entity['name']
entity.pop('name')                  # KeyError if absent
entity.pop('name', None)

del collection['name']              # from every entity in the collection
```

`__delitem__` is the natural spelling because `__setitem__` is how attributes
are written, and `pop` mirrors `dict`. The collection form is the one analysis
code needs: a derived value computed wrongly for every ROI is removed from every
ROI, not in a loop over 42,000 entities.

### Rules

- **Same transaction rules as a write.** Outside a context the row is deleted at
  once. Inside `with ent:` the name goes on a per-entity `_attributes_to_delete`
  list and the entity on `_entities_to_update`; `Entity.commit()` issues the
  deletes before the writes. A set after a queued delete cancels the delete; a
  delete after a queued set drops the set and queues the delete. `discard_pending`
  forgets both.
- **Reads see the deletion at once.** After `del entity['x']`, `'x' in entity`
  is False and `entity['x']` raises `KeyError`, committed or not. The cache
  entry is popped and the pending delete is consulted before the backend is.
- **Immutable attributes follow the overwrite rule.** An attribute written in
  digest mode is immutable, and overwriting it is refused outside digest mode.
  Removing it is refused the same way, with the same message. `id` and `uuid`
  are never removable; they mirror the entity row.
- **Absence is an error on `del`, not on `pop(name, default)`.** As for a dict.
- **Nothing is written about what was removed.** There is no audit log for
  writes either. See the open question at the end.

### Backend

```python
Backend.delete_entity_attributes(entity, names) -> list[str]   # owned files to remove
Backend.delete_collection_attribute(collection, name) -> list[str]
```

Both return the files their rows owned, so that the core removes them after the
transaction. The collection form uses the collection's query the way
`set_collection_attributes` does, in batches of `MAX_BOUND_PARAMETERS`.

## Entities

### API

```python
parent.remove_child(child)                                 # child has no children
parent.remove_child(child, recursive=True)                 # child and everything below
ent.remove_child(experiment, recursive=True)               # root level, forwards

report = parent.remove_child(child, recursive=True, dry_run=True)
```

`remove_child` sits where `add_child` sits, at every level, so the two read as a
pair and the entarchy's own `remove_child` forwards to the root entity like its
`add_child` does. A child is named by handle; removing by id would mean a second
lookup rule, and `parent.get(Type, 'id == "x"')[0]` already finds one.

The call checks that `child.parent` is this entity, the way `add_child` checks
the type against the hierarchy, so a handle from the wrong place cannot remove
anything.

### Rules

- **Children block, unless asked.** An entity with children is refused without
  `recursive=True`. The refusal names what is below: *"rec_01 has 1 Imaging, 5
  Layer, 1,208 Roi below it; pass recursive=True to remove them with it."* That
  is the difference between removing an imaging stub and removing a recording,
  and the caller should have to say which they meant.
- **Links go with their endpoints, both ways, and do not count as children.** A
  link whose linker or linked entity is gone means nothing, so links are removed
  with the entity whether or not `recursive` is set. The carrier entities sit
  under the linker as `LinkEntity` children; those under the removed entity are
  its own, those under other linkers that point at it are found by
  `linked_uuid`. The report lists them per link kind.
- **The report comes first.** Before anything is deleted the call prints what
  will go: entities per type, attributes, bytes in rows and in files, links per
  kind. With `dry_run=True` it returns that report and deletes nothing. This is
  `describe` applied to a subtree, and it reuses `count_child_entities` and
  `get_attribute_storage`.
- **Immutable entities follow the digest rule.** An entity whose attributes were
  written in digest mode is an ingested fact. Removing one outside digest mode
  is refused, as rewriting its attributes is: `ent.start_digest()`, or a
  `@digest_method` such as the vxpy ingest, says that the caller is curating
  the data rather than analysing it. An entity with only mutable attributes -
  an analysis entity, a link carrier, a stub whose ingest failed before any
  attribute was written in digest mode - is removable by anyone. *This is the
  one rule here that is a choice rather than a consequence; see the open
  questions.*
- **Pending entities need no rows.** Removing an entity that is still queued for
  insertion dequeues it and unbinds it, which is `discard_pending` for one
  entity. Its queued attribute writes and links go with it.
- **The Python object is dead afterwards.** It leaves the identity map, its
  `_entarchy` is cleared and a `_removed` flag makes any further read or write
  raise `RuntimeError(f'{self} was removed')` rather than `KeyError` or, worse,
  re-adding it through a stale cache. Collections that held it return stale
  cached values until they are re-created; `Collection` re-queries counts and
  rows but caches attribute columns, and the removal cannot find every
  collection that holds a uuid. The docstring says so.
- **One transaction per call.** The subtree is collected first - uuids by depth,
  owned files, link rows - then deleted in one `backend.batch()`, deepest first.
  A failure part way leaves nothing changed, as `commit()` promises for writes.
  Files are removed after the batch returns.

### Backend

```python
Backend.collect_subtree(uuid) -> list[tuple[str, str, int]]     # (uuid, type, depth)
Backend.remove_entities(uuids) -> list[str]                      # owned files to remove
```

`remove_entities` deletes, per batch of `MAX_BOUND_PARAMETERS`: attribute rows
of the uuids and of their link carriers; link rows by `link_uuid`, `linker_uuid`
or `linked_uuid`; carrier entity rows; the entity rows, deepest first. The
caller passes the subtree already ordered. `remove_links_of_type` becomes a
caller of the same per-batch steps.

## What is out of scope

- **Removing an entity type.** Possible only when no entity of it exists, and it
  changes what `_load_types` checks a class against. A separate change, small
  once entity removal exists, not needed by anything yet.
- **Repacking archives.** The archive tooling's concern.
- **Referential integrity inside values.** A matrix stored with uuid axes, a
  `frame_times` array a Phase link indexes into, a `zstack_fn` naming a file:
  entarchy does not know what a value refers to, and a removal cannot fix what
  it does not know about. The dense-matrices proposal should say that its axes
  can go stale, and `matrix_from_links` should drop rows for uuids that no
  longer exist rather than fail.
- **Soft deletion.** A `removed` flag on rows that queries skip would allow
  undo and keep the audit. It also means every query carries the flag, every
  unique index on `(parent_uuid, type, id)` has to allow a removed row beside a
  live one, and `describe` has two counts of everything. Not for a first
  version; the case for it is made by the first removal somebody regrets.

## Open questions

1. **Should removing an ingested entity require digest mode?** The rule above
   says yes, by analogy with immutable attributes. The counter-argument is that
   deletion is already loud - a report, a `recursive` flag - and that making the
   curating user switch modes to remove a broken recording adds a step without
   adding information. Either is defensible; the proposal picks consistency with
   the write rules, and would drop it if the switch proves to be the step
   everybody forgets.
2. **Should a removal leave a trace?** Nothing in entarchy records who wrote
   what when, beyond the `created` and `modified` columns, so a removal that
   records nothing is consistent. A one-line note in the console is what the
   report already is. If an audit becomes wanted it is wanted for writes too,
   and belongs in a proposal of its own.
3. **Is `del collection['name']` too easy?** It removes an attribute from every
   entity a query matched, in one statement. The argument for it is that
   `collection['name'] = series` writes them all the same way and nobody asked
   for a guard there. Printing the count before deleting, as the entity form
   does, is enough.

## What this would have changed today

The imaging stubs: `recording.remove_child(imaging)` per recording, eight lines
in a loop instead of a function that knows three table names. The failed
ingest itself now leaves nothing behind, since a failed write context discards
what it queued (`895e5d7`), so the stubs are a one-off - but the next wrong
analysis column, the next recording ingested from the wrong folder, and the next
`add_recording` that should replace rather than skip are not.

## Estimate

Backend `collect_subtree`, `remove_entities`, `delete_entity_attributes`,
`delete_collection_attribute` and the report query: about 200 lines in
`sql.py`, shared by both SQL backends. Core `__delitem__`, `pop`,
`remove_child` with the pending-queue handling and the dead-object flag: about
150 lines. Tests for each rule above, including the file cleanup and the
MySQL delete order: about 300 lines.
