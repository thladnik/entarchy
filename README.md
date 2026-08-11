## Entarchy - Hierarchical entity data manger for analysis prototyping

Entarchy stores hierarchically organized entities (e.g. Animal > Recording > Roi)
with arbitrary, dynamically added attributes on top of a SQL backend (SQLite or
MySQL). Scalar attributes are stored as typed columns and are queryable with
string filter expressions; arrays and other objects are stored as blobs, with
large payloads transparently written to sharded files on disk.

### Filter expressions

```python
rois = ent.get(Roi, 'has_receptive_field == True AND [Animal]strain == "wt"')
rois = ent.get(Roi, 'index IN (1, 2, 3)')
rois = ent.get(Roi, 'NOT(EXIST(dff)) OR quality != "bad"')
```

Operator precedence follows SQL/Python conventions
(comparisons > `NOT` > `AND` > `XOR` > `OR`); use parentheses to group.
Supported comparisons: `==`, `!=`, `<`, `<=`, `>`, `>=`, `IN (...)`, `EXIST(attr)`.
Parent attributes are addressed with `../attr` (one level per `../`) or
`[ParentEntityTypeName]attr`.

### Links

A link is a relationship between two entities, carried by an entity of its own,
so it holds attributes exactly as any other entity does. It expresses what the
parent hierarchy cannot: a ROI's response to a stimulation phase, the same cell
across two sessions, correlated pairs.

The *kind* of link is data rather than a class, so one can be invented at the
prompt the way an attribute name can. What it may connect is recorded in the
database and checked on every write:

```python
ent.define_link_type('mean_response', Phase, Roi,
                     description='trial-averaged dF/F during the phase')
ent.define_link_type('correlated', Roi, Roi, symmetric=True)

response = ent.link(phase, roi, 'mean_response', mean_dff=0.42)
ent.link_from_frame(df, 'mean_response')                     # df has linker_uuid/linked_uuid
ent.link_from_matrix(rois, rois, r, 'correlated',            # the predicate is required
                     where=lambda v: abs(v) > 0.6, value_name='r')
```

`symmetric` only has to be given when both endpoints are the same type;
otherwise the endpoint types already say which end is which, and arguments in
the wrong order are oriented rather than rejected.

Querying uses the ordinary filter language, with `@` addressing an endpoint:

```python
ent.links('mean_response', '@Phase.index == 3 AND @Roi.has_receptive_field == True')
ent.links('mean_response', '@linker.[Recording]imaging_rate > 8.0 AND mean_dff > 0.3')
ent.links('correlated', '@Roi.quality == "good" AND r > 0.8')
```

A bare name is an attribute of the link itself. Endpoints are addressed by
entity type, or by role: `@linker`, `@linked`, `@either` (at least one end) and
`@both`. `@linker`/`@linked` are refused for a symmetric kind, where which end
is which is an artifact of uuid ordering.

A collection can ask for the links reaching it, or the links *among* its own
members — the collection against itself:

```python
rois = ent.get(Roi, 'has_receptive_field == True')

rois.links('correlated')                 # at least one end is one of these ROIs
rois.links('correlated', within=True)    # both ends are, i.e. among them
rois.links('correlated', 'r > 0.8', within=True)

day2.links_to(day1, 'same_cell')         # between two collections
```

Membership is applied as a subquery, so it composes with whatever filter the
collection already carries and has no size limit. Spelling the members out as
`@both.uuid IN (...)` also works, but binds one parameter per uuid per endpoint
and gives up at roughly sixteen thousand.

`ent.links(...)` returns a `LinkCollection`, so `map_async`, `dataframe_of`,
`update`, set operations and `to_asdf` all work on links.

**Links are for sparse relationships.** A link costs roughly 1.5 kB against 4
bytes for a `float32`, so an all-to-all matrix belongs on the nearest common
ancestor as an array, not as links. Bulk writes refuse to be enormous or to be
most of every possible pair unless told otherwise — see
[the proposal](docs/proposals/link-entities.md) for the reasoning and the
measurements.

### Query planner statistics (SQLite)

Attributes are stored one row per attribute, with a typed value column per kind.
[The proposal](docs/proposals/attribute-storage.md) benchmarks that against
JSON-column alternatives and explains why it stays.

SQLite plans entarchy's filters badly without statistics — driving a query from
the entity type rather than from the attribute it names. The backend collects
them when it closes, so this is only for a database that is read far more than
it is written:

```sh
# dry run first, then apply
python -m entarchy.tools.optimize_storage /path/to/entarchy
python -m entarchy.tools.optimize_storage /path/to/entarchy --apply
```

MySQL is not covered: InnoDB maintains its own.

### Database credentials (MySQL backend)

The database password is **not** written to `entarchy.yaml`. At runtime it is
resolved in this order:

1. explicit `dbpassword` argument (legacy configs that still contain a
   password keep working),
2. the `ENTARCHY_DB_PASSWORD` environment variable,
3. an interactive prompt.

### How values are stored

Scalars — `str`, `int`, `float`, `bool`, `date`, `datetime` — go into a typed
column of their own, so filters compare against them directly. Strings have no
length limit.

Everything else is a blob: arrays, lists, dicts, tuples, bytes. Those are
encoded by `entarchy.backend.blob_store` into a JSON header describing the
structure plus the arrays as raw buffers, held in the row unless the result
reaches `max_blob_size` (10 MB by default), in which case it goes to a file
under `ext/` and the row keeps a pointer.

**Nothing stored is a pickle**, which matters twice over. Reading an entarchy
does not execute code from it, so a dataset from someone else is safe to open;
and the stored bytes name no Python module, so the data does not depend on
entarchy's or numpy's internal layout. Values that genuinely cannot be encoded —
custom classes, object arrays — still fall back to a pickled block, but only
that value, and the export reports every one.

Compression is decided per value and kept only when it pays, so incompressible
float traces are stored raw. On real calcium imaging data the format is about
35% smaller than the pickles it replaced.

### Archiving to ASDF

An entarchy, or a collection out of one, can be exported to a self-describing
[ASDF](https://asdf-standard.readthedocs.io) archive:

```sh
python -m entarchy.tools.archive export /path/to/entarchy /path/to/archive
```

```python
ent.to_asdf('/path/to/archive')
ent.get(Roi, 'has_receptive_field == True').to_asdf('/path/to/figure_3_data')
```

An archive **is** an entarchy directory, so analysis and figure code opens it
unchanged:

```python
ent = MyArchy('/path/to/archive')
rois = ent.get(Roi, '[Animal]strain == "wt" AND good == True')
rois.dataframe_of(['index', 'quality'])
```

    archive/
        entarchy.yaml     names entarchy.backend.archive.ArchiveBackend
        index.sqlite      queryable metadata, normal entarchy schema
        meta.asdf         the same metadata, columnar and self-describing
        blocks/*.asdf     the arrays, one file per parent group

Queries run against `index.sqlite`, which has indexes and a query planner; ASDF
has neither, and its YAML tree is parsed in full on open. Only array reads touch
ASDF, where the cost is flat regardless of file size. Exporting a collection
brings its ancestors along, so parent lookups and `[Parent]attr` filters still
resolve.

`index.sqlite` is a cache, not a second source of truth. `meta.asdf` holds
everything needed to rebuild it, which is what keeps the archive readable
without entarchy:

```sh
python -m entarchy.tools.archive rebuild /path/to/archive
```

Archives are read-only. To carry on working with the data, import it back:

```sh
python -m entarchy.tools.archive import /path/to/archive /path/to/new_entarchy
```

Values that ASDF cannot express natively — custom classes, pandas objects,
object arrays — fall back to pickled blocks, and the export prints exactly which
attributes those were. An archive containing them can only be read where those
classes are importable, so the warning is worth acting on.

Requires `asdf` (`pip install entarchy[asdf]`); everything else works without it.

### Notebooks

Runnable examples live in [`examples/`](examples):

- `01_getting_started.ipynb` — schema, entities, queries, DataFrames
- `02_parallel_analysis.ipynb` — `map_async`, worker pools, failure handling

Collections and entities render as tables in a notebook. Both representations are
deliberately cheap: they show identity and attribute *names* without loading
values, since a single entity can hold hundreds of megabytes of arrays. Use
`collection.preview()` or `collection.dataframe_of([...])` to read values.

An entity that has links also lists its link kinds and how many of each, counted
in the database rather than by loading them. The line is left out entirely when
nothing links to the entity, so an entarchy that uses no links looks unchanged.
`entity.link_counts()` returns the same thing as a dict.

**One caveat for parallel work.** Worker processes are started with `spawn`, so
they must import the function being mapped. A function defined in a notebook cell
lives in a `__main__` that workers cannot import. entarchy detects this and either

- sends the definition by value, if `cloudpickle` is installed
  (`pip install entarchy[notebook]`), or
- prints a warning and runs the work in the current process.

Either way it completes; without the check the worker pool would restart dying
workers indefinitely and the kernel would appear to hang. For production
pipelines, keep analysis functions in an importable module.

Worker processes stay alive between `map_async` calls and hold open database
connections. In a long-lived kernel, release them with:

```python
from entarchy.core.entity import shutdown_worker_pool
shutdown_worker_pool()
```

### Running tests

```sh
pip install pytest
pytest                # everything
pytest -m "not slow"  # skip multiprocessing tests
```

The MySQL backend is covered by integration tests, skipped unless a server is
configured. The user needs `CREATE` and `DROP` on schemas matching the prefix;
each test creates its own schema and drops it again:

```sh
ENTARCHY_MYSQL_HOST=localhost ENTARCHY_MYSQL_USER=me ENTARCHY_DB_PASSWORD=secret ENTARCHY_MYSQL_SCHEMA_PREFIX=entarchy_test_ pytest
```
