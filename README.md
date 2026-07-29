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

### Database credentials (MySQL backend)

The database password is **not** written to `entarchy.yaml`. At runtime it is
resolved in this order:

1. explicit `dbpassword` argument (legacy configs that still contain a
   password keep working),
2. the `ENTARCHY_DB_PASSWORD` environment variable,
3. an interactive prompt.

### Repairing blob attributes written by older versions

Versions prior to this one corrupted blob attributes written through the
*collection* path (`collection['attr'] = series` / `Collection.update()`);
reading them fails with `AttributeError: 'Series' object has no attribute
'deserialize'`. The payload is fully recoverable:

```sh
# dry run first, then apply
python -m entarchy.tools.repair_blobs /path/to/entarchy_dir
python -m entarchy.tools.repair_blobs /path/to/entarchy_dir --apply
```

The tool also accepts a SQLite file path or a full SQLAlchemy URL for
MySQL-backed entarchies.

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
