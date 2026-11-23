from __future__ import annotations

import importlib
import inspect as _inspect
import os
import pathlib
import pprint
import sys
import uuid
from typing import Any, Callable, Iterable, Type, TYPE_CHECKING, Union, cast, overload

import alive_progress
import yaml

from . import console
from . import describe
from . import links
from . import query
from .entity import (AnalysisEntity, Collection, DeferredEntityCollection,
                     EntarchyEntity, Entity, EntityT, compose_expression,
                     LinkCollection, LinkEntity)

if TYPE_CHECKING:
    import pandas as pd

    from ..backend.backend import Backend
    from ..backend.mysql import MySQLBackend
    from ..backend.sqlite import SQLiteBackend


BUILTIN_TYPE_MAP: dict[str, Type[Entity]] = {'EntarchyEntity': EntarchyEntity,
                                             'AnalysisEntity': AnalysisEntity,
                                             'LinkEntity': LinkEntity}
BUILTIN_TYPES = tuple(BUILTIN_TYPE_MAP)


def _builtin_base(entity_type: type) -> type:
    for base in (EntarchyEntity, AnalysisEntity, LinkEntity):
        if issubclass(entity_type, base):
            return base
    return Entity


def _declared_tree(entity_types: Iterable[Type[Entity]]
                   ) -> tuple[dict[str, str | None], dict[str, Type[Entity]]]:
    """Walk classes and the children they declare: name -> parent name (None for
    a root), and name -> class. Two classes of one name are an error, since
    the store knows types by name alone."""
    parents: dict[str, str | None] = {}
    classes: dict[str, Type[Entity]] = {}

    def walk(entity_type: Type[Entity], parent: str | None):
        name = entity_type.__name__
        if name in classes:
            if classes[name] is entity_type and parents[name] == parent:
                return
            raise ValueError(f'Entity type {name} is already in the hierarchy. '
                             f'Circular reference or duplicate name? Entity names must be unique.')
        if name in BUILTIN_TYPES:
            raise ValueError(f'{name} is a built-in type and cannot be declared.')
        classes[name] = entity_type
        parents[name] = parent
        for child in entity_type.get_child_entity_types() or ():
            walk(child, name)

    for root in entity_types:
        walk(root, None)

    return parents, classes


def _generated_type(name: str) -> Type[Entity]:
    """A plain Entity subclass for a stored type no class was bound to."""
    return type(name, (Entity,), {'__module__': __name__, '_generated': True,
                                  '__doc__': f'Stored entity type {name}; no class was bound to it.'})


class Entarchy(object):
    """Entarchy is a class that represents a system of hierarchically organized entities.
    """
    # 0.2 changed how attribute values are stored: blobs hold an encoded
    #  container rather than a pickled object, and value_str is unbounded. A 0.1
    #  directory cannot be read, and is refused here rather than failing later
    #  inside a value read.
    _base_version: str = '0.2'
    _base_compat_version_list: list[str] = ['0.2']
    # The entity classes this entarchy binds to the stored type hierarchy, by
    #  name. Children declared on them with add_child_entity_type are followed,
    #  so naming the roots is enough. The hierarchy itself lives in the store:
    #  these only say which Python class an entity of a stored type becomes.
    _entity_types: tuple[Type[Entity], ...] = ()
    max_blob_size: int = 10 * 1024 * 1024  # bytes (10MB)

    # TODO: Make this more generic, currently only MySQLBackend is supported:
    _backend: MySQLBackend | SQLiteBackend = None  # Backend
    _config: dict[str, Any] = None
    _is_in_context: bool = False
    _is_in_digest_mode: bool = False
    _current_analysis: Union[AnalysisEntity, None] = None
    _entarchy_entity: EntarchyEntity = None

    def __init__(self, path: str, entity_types: Iterable[Type[Entity]] = (),
                 debug: bool = False, _defer_types: bool = False):
        """Open the entarchy at `path`.

        `entity_types` binds further classes to stored types, beyond the ones
        the class declares in `_entity_types`. A stored type with no class bound
        to it is still usable: it gets a plain Entity subclass of that name.
        """
        self._path = pathlib.Path(path).absolute().as_posix()
        self._debug = debug

        self._bound_types: tuple[Type[Entity], ...] = tuple(self._entity_types) + tuple(entity_types)
        # Read from the store on first use (or right below), never from the classes
        self._type_parents: dict[str, str | None] | None = None
        self._entity_map: dict[str, Type[Entity]] | None = None

        # Load configuration from path
        self._config = yaml.safe_load(open(os.path.join(path, 'entarchy.yaml'), 'r'))

        if self._config['base_version'] not in self._base_compat_version_list:
            raise RuntimeError(
                f'"{self._path}" was written by entarchy base version '
                f'{self._config["base_version"]}; this is {self._base_version}, which '
                f'reads {", ".join(self._base_compat_version_list)}. The storage format '
                f'changed between them, so the data has to be regenerated.')

        # Set up entities objects
        self._entities: dict[str, Entity] = {}
        self._entities_to_add: list[str] = []
        self._entities_to_update: list[str] = []
        # Link rows wait for their carrier entities to exist, so they are queued
        #  and inserted between the entity insert and the attribute writes.
        #  The index lets a second link() for the same pair inside one block find
        #  the first, which is not yet in the database to be queried for.
        self._links_to_add: list[dict] = []
        self._pending_link_index: dict[tuple, str] = {}

        # create() opens the store before its tables exist and loads afterwards
        if not _defer_types:
            self._load_types()

    def __contains__(self, item):
        if isinstance(item, Entity):
            return item.uuid in self._entities
        elif isinstance(item, str):
            return item in self._entities
        return False

    def __enter__(self):
        self._is_in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Only commit if the context is left without an exception,
        #  otherwise partially processed state would be persisted
        if exc_type is None:
            self.commit()
        self._is_in_context = False

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove backend from serialization, so it may stay open on original object
        #  This prevents broken file handles or database connections
        state.pop('_backend', None)
        # Do not ship the entity registry or pending commit queues to subprocesses.
        #  The registry grows with every entity ever loaded, which makes pickling
        #  each map_async task O(total entities) instead of O(1).
        state['_entities'] = {}
        state['_entities_to_add'] = []
        state['_entities_to_update'] = []
        state['_links_to_add'] = []
        state['_pending_link_index'] = {}
        # The type map may hold classes generated for stored types nobody bound,
        #  which pickle cannot name. The bound classes travel, and a worker reads
        #  the hierarchy again from its own connection on first use
        state['_type_parents'] = None
        state['_entity_map'] = None
        return state

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return f'{self.__class__.__name__}(\'{self.path}\', backend={self.backend.__class__.__qualname__})'

    @property
    def backend(self) -> MySQLBackend | SQLiteBackend:  # Backend:

        if not hasattr(self, '_backend') or self._backend is None:
            # Load backend
            _backend_path = self._config['backend']
            _backend_path_parts = _backend_path.split('.')
            _backend_cls = getattr(importlib.import_module('.'.join(_backend_path_parts[:-1])), _backend_path_parts[-1])
            self._backend = _backend_cls(self.path, **self._config['backend_config'], debug=self._debug)

            # Keep the resolved runtime config (e.g. interactively provided credentials,
            #  which are intentionally not persisted to entarchy.yaml) in memory,
            #  so worker subprocesses don't have to prompt again
            self._config['backend_config'] = self._backend.get_runtime_config()

        return self._backend

    @property
    def base_version(self):
        return self._base_version

    @property
    def debug(self) -> bool:
        """Get the debug mode for the entarchy and its backend.
        """
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        """Set the debug mode for the entarchy and its backend.
        """
        self._debug = value
        self._backend.debug = value

    @property
    def hierarchy(self) -> dict[str, Any]:
        """The stored entity type hierarchy as nested dicts, roots at the top.

        The three built-in types come first, each a root of its own; the
        user's types follow in the order they were added.
        """
        parents = self._parents()
        nested: dict[str, Any] = {}
        branches = {None: nested}
        for name, parent in parents.items():
            branches[name] = branches[parent][name] = {}
        return nested

    @property
    def entity_types(self) -> tuple[Type[Entity], ...]:
        """The classes of the stored types, bound or generated, roots first."""
        return tuple(self._type_map()[name] for name in self._parents()
                     if name not in BUILTIN_TYPES)

    def child_type_names(self, parent: str | None) -> list[str]:
        """Names of the stored types directly under `parent`; the user's root
        types for None."""
        return [name for name, parent_name in self._parents().items()
                if parent_name == parent and name not in BUILTIN_TYPES]

    def add_child_entity_type(self, entity_type: Type[Entity] | str) -> Type[Entity]:
        """Add a root entity type, and return its class.

            Subject = ent.add_child_entity_type('Subject')
            ent.add_child_entity_type(Experiment)    # a class, with the children
                                                     #  declared on it

        The root types are the children of the root entity, so this is the
        entarchy's side of `Entity.add_child_entity_type`: on a class that
        declares a child type, here it is written into the store. It is
        `add_entity_type(entity_type)` with no parent; `add_entity_type` with
        `parent=` puts a type under a stored type by name.
        """
        return self.add_entity_type(entity_type, parent=None)

    def add_entity_type(self, entity_type: Type[Entity] | str,
                        parent: Type[Entity] | str | None = None) -> Type[Entity]:
        """Add an entity type to the store, under `parent` or as a root, and return
        its class.

            Subject = ent.add_entity_type('Subject')
            Session = ent.add_entity_type('Session', parent=Subject)
            ent.add_entity_type(Experiment)      # a class, with the children
                                                 #  declared on it

        A class is bound to the type of its name; a string gets a generated
        plain Entity subclass. Given a class, the child types it declares with
        add_child_entity_type go in under it as well, recursively, so one call
        adds a whole declared schema to a store that grew up without it.

        Adding a type that is already there, under the same parent, changes
        nothing and returns its class. Under a different parent it is an error:
        types are only ever added, never moved, renamed or removed.
        """
        parents = self._parents()
        type_map = self._type_map()

        if isinstance(entity_type, str):
            name, declared = entity_type, None
        elif isinstance(entity_type, type) and issubclass(entity_type, Entity):
            if issubclass(entity_type, (EntarchyEntity, AnalysisEntity, LinkEntity)):
                raise TypeError(f'{entity_type.__name__} cannot be an entity type of the '
                                f'hierarchy; it is {_builtin_base(entity_type).__name__}.')
            name, declared = entity_type.__name__, entity_type
        else:
            raise TypeError(f'entity_type must be an Entity subclass or a name, '
                            f'not {type(entity_type).__name__}.')

        if name in BUILTIN_TYPES:
            raise ValueError(f'{name} is a built-in type.')

        if parent is None:
            parent_name = None
        else:
            parent_name = parent if isinstance(parent, str) else parent.__name__
            if parent_name not in parents or parent_name in BUILTIN_TYPES:
                raise ValueError(f'Parent type {parent_name} is not in the hierarchy; '
                                 f'add it first.')

        if name in parents:
            if parents[name] != parent_name:
                raise ValueError(
                    f'{name} is already in the hierarchy under '
                    f'{parents[name] or "the root"}, not {parent_name or "the root"}. '
                    f'Types are added, never moved.')
            if declared is not None and type_map[name] is not declared \
                    and not getattr(type_map[name], '_generated', False):
                raise ValueError(f'{name} is already bound to {type_map[name]!r}.')
        else:
            self.backend.add_entity_type(name, parent_name)
            parents[name] = parent_name

        if declared is not None:
            type_map[name] = declared
        elif name not in type_map:
            type_map[name] = _generated_type(name)

        # The children a class declares come with it
        for child in (declared.get_child_entity_types() or ()) if declared else ():
            self.add_entity_type(child, parent=declared)

        return type_map[name]

    def _parents(self) -> dict[str, str | None]:
        if self._type_parents is None:
            self._load_types()
        return self._type_parents

    def _type_map(self) -> dict[str, Type[Entity]]:
        if self._entity_map is None:
            self._load_types()
        return self._entity_map

    def _load_types(self) -> None:
        """Read the hierarchy from the store and bind the declared classes to it.

        A class declared here but missing from the store is an error, with the
        call that adds it. A stored type with no class gets a generated one.
        """
        stored = self.backend.get_type_hierarchy()
        parents: dict[str, str | None] = {name: None for name in BUILTIN_TYPES}
        parents.update({name: parent for name, parent in stored if name not in BUILTIN_TYPES})

        declared_parents, declared_map = _declared_tree(self._bound_types)

        for name, declared_parent in declared_parents.items():
            if name not in parents:
                where = f'parent={declared_parent}' if declared_parent else 'as a root'
                raise RuntimeError(
                    f'{type(self).__name__} declares an entity type {name} that '
                    f'"{self._path}" does not have. Add it with '
                    f'ent.add_entity_type({name}) ({where}), or open the entarchy '
                    f'without that class.')
            if parents[name] != declared_parent:
                raise RuntimeError(
                    f'{type(self).__name__} declares {name} under '
                    f'{declared_parent or "the root"}, but in "{self._path}" it is '
                    f'under {parents[name] or "the root"}. Types are never moved; '
                    f'this entarchy was built with a different schema.')

        type_map: dict[str, Type[Entity]] = dict(BUILTIN_TYPE_MAP)
        for name in parents:
            if name in type_map:
                continue
            type_map[name] = declared_map.get(name) or _generated_type(name)

        self._type_parents = parents
        self._entity_map = type_map

    @property
    def is_in_context(self) -> bool:
        """Check if the entarchy is in a context manager.

        Returns:
            bool: True if the entarchy is in a context manager, False otherwise.
        """
        return self._is_in_context

    @property
    def is_in_digest_mode(self):
        return self._is_in_digest_mode

    @property
    def path(self) -> str:
        return self._path

    @property
    def root(self):

        if self._entarchy_entity is None:

            # Load entarchy entity
            entarchy_uuid = self._config.get('entarchy_uuid')
            if entarchy_uuid is None:
                raise RuntimeError('No entarchy UUID found in configuration. Is the entarchy initialized correctly?')

            self._entarchy_entity = self.backend.get_entity_by_uuid(self, entarchy_uuid)

        return self._entarchy_entity

    @classmethod
    def _declared_hierarchy(cls, entity_types: Iterable[Type[Entity]] = ()):
        """The hierarchy the classes declare, as create() will store it: the
        nested dict and the name -> class map, built-in types included."""
        parents, declared_map = _declared_tree(tuple(cls._entity_types) + tuple(entity_types))

        hierarchy: dict[str, Any] = {name: {} for name in BUILTIN_TYPES}
        branches = {None: hierarchy}
        for name, parent in parents.items():
            branches[name] = branches[parent][name] = {}

        entity_map = dict(BUILTIN_TYPE_MAP)
        entity_map.update(declared_map)

        return hierarchy, entity_map

    @classmethod
    def open_or_create(cls,
                       path: str,
                       _backend: Backend | str = 'SQLiteBackend',
                       _backend_config: dict[str, Any] = None,
                       **kwargs) -> Entarchy:
        """Open the entarchy at `path`, or create one there if there is none.

        An ingest that may be run twice needs this: create() refuses a path that
        exists and the constructor refuses one that does not, so every caller
        who wanted "continue where it stopped" had to test for entarchy.yaml
        itself.

            ent = LabArchy.open_or_create(path)

        The backend arguments are only used when creating, and default as in
        create(); opening reads them back from the configuration that was
        written then.
        """
        if os.path.exists(os.path.join(path, 'entarchy.yaml')):
            return cls(path, **kwargs)

        return cls.create(path, _backend, _backend_config, **kwargs)

    @classmethod
    def create(cls,
               path: str,
               _backend: Backend | str = 'SQLiteBackend',
               _backend_config: dict[str, Any] = None,
               entity_types: Iterable[Type[Entity]] = (),
               **kwargs) -> Entarchy:
        """Create an entarchy at `path` and return it.

            ent = Entarchy.create('path/to/folder')

        is a complete call: without a backend it is an SQLite database,
        `entarchy.db`, in that folder. `'MySQLBackend'` with its config
        dictionary, or a Backend instance, puts it elsewhere.

        The store starts with the types the class declares in `_entity_types`,
        plus `entity_types`, each with the children declared on it. A plain
        `Entarchy.create(...)` starts with none; types are then added with
        `add_entity_type` as the entarchy grows.

        Args:
            path (str): The path to the entarchy configuration directory.
            _backend (Backend): an entarchy.backend.Backend object or its class name;
                SQLiteBackend if not given
            _backend_config (dict): dictionary for initialization of backend if _backend is str or type;
                for SQLite, {'dbname': ...} with 'entarchy.db' the default
            entity_types: classes to bind and store, beyond `_entity_types`
        Returns:
            Entarchy:
        """

        # Instantiate backend if necessary
        if isinstance(_backend, str):
            if '.' in _backend:
                _backend_path_parts = _backend.split('.')
                _backend_cls = getattr(importlib.import_module('.'.join(_backend_path_parts[:-1])), _backend_path_parts[-1])
            else:
                from .. import backend
                _backend_cls = getattr(backend, _backend)

            _backend = _backend_cls(path, **(_backend_config or {}), debug=False)

        # What the classes declare is what the store starts with
        hierarchy, entity_map = cls._declared_hierarchy(entity_types)

        # An existing path is refused because creating into it would write over
        #  whatever is there. An empty directory holds nothing to write over, and
        #  having made the destination first - by hand, or as a mount point - is
        #  ordinary enough that refusing it only invites a rmdir.
        if os.path.exists(path):
            if not os.path.isdir(path) or len(os.listdir(path)) > 0:
                raise FileExistsError(f'Path {path} already exists.')
        else:
            os.makedirs(path, exist_ok=False)

        print('---')
        print('Create entity type hierarchy:')
        pprint.pprint(hierarchy)
        print('---')

        # Save configuration so entarchy object can be created
        _config = {
            'base_version': cls._base_version,
            'backend': f'{_backend.__module__}.{_backend.__class__.__name__}',
            'backend_config': _backend.get_config(),
        }

        with open(os.path.join(path, 'entarchy.yaml'), 'w') as f:
            yaml.safe_dump(_config, f)

        # Create instance. The type hierarchy is read from the store, which does
        #  not exist yet, so that read is deferred until it has been written
        ent = cls(path, entity_types=entity_types, _defer_types=True, **kwargs)

        # Reuse the backend instance that was already configured above,
        #  so credentials are not requested a second time
        ent._backend = _backend
        # Keep the runtime config (incl. values that are not persisted, such as the
        #  database password) in memory, so worker processes can connect without prompting
        ent._config['backend_config'] = _backend.get_runtime_config()

        # Create backend
        res = ent.backend.create()
        if not res:
            raise RuntimeError('Failed to create backend.')

        # Create type hierarchy in backend
        res = ent.backend.create_type_hierarchy(hierarchy)
        if not res:
            raise RuntimeError('Failed to create entity type hierarchy in backend.')
        ent._load_types()

        # Create entarchy entity
        with ent:
            ent_entity = EntarchyEntity('Entarchy', _entarchy=ent)
            ent._add_entity(ent_entity)

        # Update config to include entarchy entity uuid
        _config.update({'entarchy_uuid': ent_entity.uuid})
        with open(os.path.join(path, 'entarchy.yaml'), 'w') as f:
            yaml.safe_dump(_config, f)

        # Keep the in-memory config of the returned instance in sync with the file,
        #  otherwise ent.root fails until the entarchy is reopened from disk
        ent._config['entarchy_uuid'] = ent_entity.uuid

        return ent

    def _add_entity_for_update(self, entity: Entity) -> None:
        """Mark an entity for update in the backend.

        Args:
            entity (Entity): The entity to mark for update.

        Returns:
            None
        """

        if entity.uuid not in self._entities_to_update:
            self._entities_to_update.append(entity.uuid)

    def add_child(self, child: Type[EntityT] | EntityT, _id: str | None = None, /,
                  **attributes: Any) -> EntityT:
        """Add an entity of a root type, and return it.

            subject = ent.add_child(Subject, 'subject01', strain='wt')

            subject = Subject('subject01')
            subject['strain'] = 'wt'
            ent.add_child(subject)

        Entities of the root types are the children of the root entity, so this
        is `ent.root.add_child(...)`: the same call at every level of the tree,
        with the same type check. See `Entity.add_child`.
        """
        return self.root.add_child(child, _id, **attributes)

    def _add_entity(self, entity: Entity) -> None:
        """Store a new entity, bound to this entarchy.

        What `add_child` calls once the parent is set and the type checked.
        Outside a `with ent:` block the entity is written at once, with the
        attributes set on it so far; inside, on exit. The built-in types come
        through here parentless.
        """

        if not isinstance(entity, Entity):
            hint = ''
            if isinstance(entity, DeferredEntityCollection):
                hint = (f' {entity!r} is a deferred query; .get_from(ent) evaluates it, '
                        f'and parent.add_child({entity._entity_type.__name__}, "id") '
                        f'creates an entity.')
            raise TypeError(f'_add_entity takes an Entity, not '
                            f'{type(entity).__name__}.{hint}')

        if not entity.is_bound:
            entity._entarchy = self
        elif entity.entarchy is not self:
            raise ValueError(f'{entity} belongs to a different entarchy.')

        # An entity of a root type with no parent goes under the root entity,
        #  which is where add_child puts them; a child type has to say whose
        #  child it is. The built-in types stay parentless.
        type_name = type(entity).__name__
        if entity._parent is None and type_name not in BUILTIN_TYPES:
            parent_type = self._parents().get(type_name)
            if parent_type is None:
                entity._parent = self.root
            else:
                raise ValueError(f'{entity} needs a parent: {type_name} is a child type of '
                                 f'{parent_type}. parent.add_child(entity) sets it.')

        # Queue the entity first, so attribute writes below are deferred and
        #  the entity row is guaranteed to exist before its attribute rows
        self._register_entity(entity)
        self._entities_to_add.append(entity.uuid)

        # Attributes written while the entity was unbound go in with it
        if len(entity._attributes_to_update) > 0:
            self._add_entity_for_update(entity)

        # Add immutable id attributes
        entity['id'] = entity.id
        entity['uuid'] = entity.uuid

        # Outside a context, persist immediately (entities first, then attributes)
        if not self.is_in_context:
            self.commit()

    def is_pending_add(self, entity: Entity) -> bool:
        """Check whether an entity is queued for insertion but not yet committed."""
        return entity.uuid in self._entities_to_add

    def _register_entity(self, entity: Entity) -> None:
        """Put an entity in the identity map, so that materializing its uuid
        again gives this same object."""
        self._entities[entity.uuid] = entity

    def commit(self) -> None:
        """Commit new entities to the backend.

        Returns:
            None
        """

        # One transaction for the whole commit. Committing per entity meant one
        #  fsync each, which dominated everything else - 5.91 ms against 0.028 ms
        #  for the same write inside a batch. It also means a failure part way
        #  through leaves nothing behind rather than half the entities.
        with self.backend.batch():

            # Add new entities
            if len(self._entities_to_add) > 0:

                res = self.backend.add_entities([self._entities[_uuid] for _uuid in self._entities_to_add])
                if not res:
                    raise RuntimeError('Failed to add new entities to backend.')

                print(f'Added {len(self._entities_to_add)} entities')

                # Reset list
                self._entities_to_add = []

            # Link rows reference their carrier entity, so they follow the insert
            #  above and precede the attribute writes that hang off them
            if len(self._links_to_add) > 0:
                self.backend.add_links(self._links_to_add)
                self._links_to_add = []
                self._pending_link_index = {}

            # Commit updates for entities with attribute changes
            #  Note to future self: USE COPY, otherwise iterator is going to
            #  skip entries as tthe length of the list changes while updaed elements are removed
            _entities_to_update = self._entities_to_update.copy()
            with alive_progress.alive_bar(monitor=f'| Update {len(_entities_to_update)} entities',
                                          monitor_end=f'Updated {len(_entities_to_update)} entities',
                                          stats=False, force_tty=True,
                                          **console.bar_style(bar=None)) as bar:
                for _uuid in _entities_to_update:
                    self._entities[_uuid].commit()
                    bar()

    @property
    def current_analysis(self) -> Union[AnalysisEntity, None]:
        """Get the current analysis for the entarchy system.

        Returns:
            Union[AnalysisEntity, None]: The current analysis, or None if no analysis is set.
        """
        return self._current_analysis

    def delete(self):

        # Convert
        path = pathlib.Path(self.path).as_posix()

        import random
        import string

        verification_str = ''.join(random.choices(string.ascii_letters + string.digits, k=5))
        print('-----\nCAUTION - DANGER ZONE\n-----')
        print(f'Are you sure you want to delete the entity hierarchy on path')
        print(f'"{path}"')
        print('This means that ALL DATA WILL BE LOST!')
        print(f'If you are sure, type "{verification_str}" to verify')
        print('-----\nCAUTION - DANGER ZONE\n-----')
        user_input = input('Verification string: ')

        if verification_str != user_input:
            print('Abort.')
            return

        print(f'Deleting all data for analysis {path}')

        print('> Remove backend')
        self.backend.delete(True)

        # Release open connections/file handles before removing files
        #  (an open SQLite database file cannot be deleted on Windows)
        self.backend.close()

        print('> Remove directories and files')

        def rmdir(directory, counter):
            # Use pathlib recursive unlinker by mitch from https://stackoverflow.com/a/49782093
            directory = pathlib.Path(directory)
            for item in directory.iterdir():
                if item.is_dir():
                    counter = rmdir(item, counter)
                else:
                    item.unlink()

            if counter % 10 == 0:
                # print(' ' * 500, end='\n')
                sys.stdout.write('\r' + ' ' * 255)
                sys.stdout.write(f'\rRemove {directory.as_posix()}')
            counter += 1
            directory.rmdir()

            return counter

        # Delete tree
        rmdir(path, 0)

        print(f'\nSuccessfully deleted analysis {path}')

    def describe(self, largest: int = describe.STORAGE_ROWS) -> describe.Description:
        """What is in this entarchy at all.

            ent.describe()                  # renders whole
            ent.describe().entities         # a DataFrame

        The question a stranger to an entarchy asks first, and the one nothing
        else here answers: which entity types are in it and how many of each,
        which link kinds join them, and where the bytes have gone. Every other
        `describe()` needs you to know what to ask about before you can ask.

        The counts ride indexes. Totalling bytes cannot - there is no adding up
        what has not been looked at - so the storage section scans the
        attributes table once, grouped down to a row per attribute name per
        entity type. That is the cost of the whole call; it does not grow with
        how much data those attributes hold, only with how many rows there are.

        A value is counted where it actually is, so a media file weighs what
        it weighs on disk rather than what its pointer row weighs. `stored` is
        therefore the size of the entarchy rather than the size of its
        database.

        Args:
            largest: how many rows of the storage section to show, largest
                first. None for all of them.

        Returns:
            Description: sections that are DataFrames, rendering together.
        """
        notes = []

        try:
            counts = self.backend.count_entities_by_type()
        except Exception as exc:
            counts = {}
            notes.append(f'entity counts could not be read: {exc}')

        try:
            link_totals = self.backend.get_link_type_totals()
        except Exception as exc:
            link_totals = {}
            notes.append(f'link counts could not be read: {exc}')

        try:
            specs = self.link_types()
        except Exception as exc:
            specs = []
            notes.append(f'link kinds could not be read: {exc}')

        try:
            storage = self.backend.get_attribute_storage()
        except Exception as exc:
            storage = []
            notes.append(f'stored sizes could not be read: {exc}')

        sizes = {}
        for type_name, _, _, _, total in storage:
            sizes[type_name] = sizes.get(type_name, 0) + total

        storage_frame, cut = describe.storage_rows(storage, limit=largest)
        if cut > 0:
            notes.append(f'{cut} smaller attribute(s) not shown. '
                         f'Pass largest= for more, or None for all of them.')

        # The entity count leaves out link carriers and the link count picks
        #  them up, so that the two sections below between them account for
        #  every byte in `stored` exactly once
        headline = {
            'path': self.path,
            'backend': type(self.backend).__name__,
            'entities': sum(count for name, count in counts.items()
                            if name != describe.LINK_ENTITY_TYPE),
            'links': sum(total['links'] for total in link_totals.values()),
            'stored': describe.human_bytes(sum(sizes.values())),
        }

        sections = {
            'entities': describe.entity_type_rows(counts, sizes, self.hierarchy),
            'links': describe.link_type_rows(specs, link_totals),
            'storage': storage_frame,
        }

        return describe.Description(
            subject=os.path.basename(str(self.path).rstrip('/\\')) or str(self.path),
            headline=headline, sections=sections, notes=notes)

    @overload
    def get(self, entity_type: Type[EntityT], *_string_expressions: str,
            **_equalities) -> Collection[EntityT]: ...

    @overload
    def get(self, entity_type: str, *_string_expressions: str,
            **_equalities) -> Collection[Entity]: ...

    def get(self, entity_type: Type[Entity] | str, *_string_expressions: str, **_equalities) -> Collection:
        """Get a collection of entities of a given type.

        Args:
            entity_type (Type[Entity]): The type of entities to get.
            *_string_expressions (str): Optional filter expressions to filter the entities
                                            (strings are concatenated with AND).
            **_equalities: Optional equality filters to filter the entities
                                            (equalities are concatenated with AND).

        Returns:
            Collection: A collection of entities of the given type.

        """

        if isinstance(entity_type, str):
            entity_type = self.get_entity_type(entity_type)

        as_tree = self._build_filter_tree(_string_expressions, _equalities)

        # Return collection of custom type if available
        if entity_type.collection_type is not None:
            return entity_type.collection_type(self, entity_type, as_tree)

        # Fallback to generic collection
        return Collection(self, entity_type, as_tree)

    def where(self, entity_type: Type[EntityT] | str, *_string_expressions: str,
              **_equalities) -> Collection[EntityT]:
        """The same as `get`: the entities of a type that match the filters.

        `where` is the one verb for filtering everywhere - `Roi.where(...)`,
        `collection.where(...)`, `ent.where(Roi, ...)` - and `get` stays for
        fetching a type whole, `ent.get(Roi)`, where nothing is filtered.
        """
        return self.get(entity_type, *_string_expressions, **_equalities)

    def _build_filter_tree(self, _string_expressions, _equalities) -> dict[str, Any]:
        """Turn filter expressions and keyword equalities into one AST."""
        as_tree = query.parse_boolean_expression(
            compose_expression(_string_expressions, _equalities))

        return {} if as_tree is None else as_tree

    def get_config(self) -> dict[str, Any]:
        return self._config.copy()

    def get_entity(self, entity_type_name: str, _uuid: str, _id: str) -> Entity:

        return self.get_entity_type(entity_type_name)(_entarchy=self, _uuid=_uuid, _id=_id)

    def get_entity_by_uuid(self, _uuid: str) -> Entity:

        if _uuid in self._entities:
            return self._entities[_uuid]
        else:
            _entity = self.backend.get_entity_by_uuid(self, _uuid)

            return _entity

    def _link_entity(self, _uuid: str) -> LinkEntity:
        """The carrier entity of a link, by uuid.

        get_entity() is given a type *name*, so nothing about the call says
        what comes back. Asking for 'LinkEntity' and saying so is knowledge
        that exists nowhere else; it lives here rather than at each of the
        three call sites that need it.
        """
        return cast(LinkEntity, self.get_entity('LinkEntity', _uuid, None))

    def get_entity_type(self, entity_type_name: str) -> Type[Entity]:
        """Get the entity type class for a given entity type name.

        Args:
            entity_type_name (str): The name of the entity type.

        Returns:
            Type[Entity]: The entity type class.
        """

        # Get type from map
        entity_type = self._type_map().get(entity_type_name, None)

        if entity_type is None:
            raise ValueError(f'Entity type {entity_type_name} not found in hierarchy.')

        return entity_type

    # def get_link(self, linker: Entity, linked: Entity) -> Union[LinkEntity, None]:
    #
    #     """Get a link entity between two entities if it exists.
    #
    #     Args:
    #         linker (Entity): The linking entity.
    #         linked (Entity): The linked entity.
    #     """
    #     raise NotImplementedError('Links are not implemented yet.')
    #
    #     link_key = (linker.uuid, linked.uuid)
    #     if link_key not in self._links:
    #         self.backend.get_link_uuid(linker, linked)
    #
    #     link_uuid = self._links[link_key]
    #     return self.get_entity('LinkEntity', link_uuid, None)

    def get_temp_path(self, path: str):
        temp_path = os.path.join(self.path, 'temp', path)
        if not os.path.exists(temp_path):
            # Avoid error if concurrent process already created it in meantime
            os.makedirs(temp_path, exist_ok=True)

        return temp_path

    def clear_registry(self) -> int:
        """Drop cached entities from the in-memory registry to free memory.

        Every entity ever loaded is kept in a registry, so that two lookups of the
        same UUID return the same object. In a long-running session (a notebook
        left open, or a script sweeping a large collection) this accumulates, along
        with whatever array data those entities have cached.

        Entities with uncommitted changes are kept, so nothing is ever lost.
        Released entities stay usable and are reloaded on demand.

        Returns:
            int: The number of entities released.
        """

        pending = set(self._entities_to_add) | set(self._entities_to_update)

        released = [_uuid for _uuid in self._entities if _uuid not in pending]
        for _uuid in released:
            del self._entities[_uuid]

        if len(pending) > 0:
            print(f'Released {len(released)} entities, '
                  f'kept {len(pending)} with uncommitted changes')

        return len(released)

    def define_link_type(self,
                         name: str,
                         linker: Any = None,
                         linked: Any = None,
                         symmetric: bool = None,
                         cardinality: str = links.DEFAULT_CARDINALITY,
                         description: str = None) -> links.LinkTypeSpec:
        """Register a kind of link and what it may connect.

        Endpoints are given as entity classes, entity type names, the name of an
        already registered link type (for a link between links), or None for a
        deliberate wildcard.

            ent.define_link_type('mean_response', Phase, Roi,
                                 description='trial-averaged dF/F during the phase')
            ent.define_link_type('correlated', Roi, Roi, symmetric=True)
            ent.define_link_type('adaptation', 'mean_response', 'mean_response')

        `symmetric` only has to be given when both endpoints are the same, since
        otherwise the endpoint types already say which end is which.

        Returns:
            links.LinkTypeSpec: the registered kind.
        """
        entity_type_names = set(self._type_map())
        link_type_names = {spec.name for spec in self.backend.get_link_types()}

        linker_endpoint = links.resolve_endpoint(linker, entity_type_names, link_type_names)
        linked_endpoint = links.resolve_endpoint(linked, entity_type_names, link_type_names)

        if symmetric is None:
            if links.requires_direction_declaration(linker_endpoint, linked_endpoint):
                raise links.LinkTypeError(
                    f'Link type "{name}" connects {linker_endpoint} to {linked_endpoint}, '
                    f'so the endpoint types cannot say which end is which. Pass '
                    f'symmetric=True for an undirected relationship, or symmetric=False '
                    f'to keep the two directions distinct.')
            symmetric = False

        spec = links.LinkTypeSpec(name=name,
                                  linker=linker_endpoint,
                                  linked=linked_endpoint,
                                  symmetric=symmetric,
                                  cardinality=cardinality,
                                  description=description)

        return self.backend.add_link_type(spec)

    def _endpoint_of(self, entity: Entity) -> links.Endpoint:
        """What an entity is, for the purposes of a link constraint."""
        if isinstance(entity, LinkEntity):
            return links.Endpoint(link_type=entity.link_type)

        return links.Endpoint(entity_type=type(entity).__name__)

    def _link_spec_for_write(self, name: str, linker: links.Endpoint,
                             linked: links.Endpoint) -> links.LinkTypeSpec:
        """The kind, registering it from the endpoints if this is its first use."""
        spec = self.backend.get_link_type(name)
        if spec is not None:
            return spec

        if links.requires_direction_declaration(linker, linked):
            raise links.LinkTypeError(
                f'Link type "{name}" has not been defined, and both endpoints are '
                f'{linker}, so its direction cannot be inferred. Define it first:\n'
                f'    ent.define_link_type({name!r}, ..., symmetric=True)   # undirected\n'
                f'    ent.define_link_type({name!r}, ..., symmetric=False)  # directed')

        return self.define_link_type(name,
                                     linker.link_type or linker.entity_type,
                                     linked.link_type or linked.entity_type)

    def link(self, linker: Entity, linked: Entity, link_type: str,
             **attributes) -> LinkEntity:
        """Connect two entities, and return the link that carries their data.

        The kind is registered on first use from the endpoints it is given,
        unless both ends are the same, where the direction has to be declared.

            response = ent.link(phase, roi, 'mean_response')
            response['mean_dff'] = 0.42

        Getting the arguments the wrong way round is fine where the kind's
        endpoint types differ, since there is then only one way it can be meant.

        Calling this again for the same pair and kind returns the existing link
        rather than failing, so re-running an analysis script is harmless.

        Returns:
            LinkEntity: the link, new or already present.
        """
        linker_endpoint = self._endpoint_of(linker)
        linked_endpoint = self._endpoint_of(linked)

        spec = self._link_spec_for_write(link_type, linker_endpoint, linked_endpoint)

        if links.orientation(spec, linker_endpoint, linked_endpoint) == 'swapped':
            linker, linked = linked, linker

        linker_uuid, linked_uuid = links.canonical_pair(spec, linker.uuid, linked.uuid)

        existing = self._existing_link_uuid(spec.name, linker_uuid, linked_uuid)
        if existing is not None:
            link_entity = self._link_entity(existing)
            if attributes:
                link_entity.update(attributes)
            return link_entity

        if spec.cardinality == 'one_per_linker':
            present = self.backend.count_links_per_linker(spec.name, [linker_uuid])
            if present.get(linker_uuid, 0) > 0:
                raise links.LinkCardinalityError(
                    f'"{spec.name}" is declared one_per_linker and {linker_uuid} already '
                    f'has one. Remove it first, or declare the kind sparse.')

        link_entity = LinkEntity(f'{spec.name}@{linked_uuid}', _entarchy=self,
                                 _parent=self.get_entity_by_uuid(linker_uuid))
        self._add_entity(link_entity)

        self._links_to_add.append({'link_uuid': link_entity.uuid,
                                   'link_type': spec.name,
                                   'linker_uuid': linker_uuid,
                                   'linked_uuid': linked_uuid})
        self._pending_link_index[(spec.name, linker_uuid, linked_uuid)] = link_entity.uuid

        if attributes:
            link_entity.update(attributes)

        # Outside a context the entity was already committed by add_entity,
        #  so the link row has to follow immediately
        if not self.is_in_context:
            self.commit()

        return link_entity

    def _existing_link_uuid(self, link_type: str, linker_uuid: str,
                            linked_uuid: str) -> Union[str, None]:
        """An existing link's uuid, whether committed or still queued.

        Checking only the database would miss a link created earlier in the same
        block, and creating a second carrier for the same pair violates the
        uniqueness of (parent, entity type, id) on the entities table.
        """
        pending = self._pending_link_index.get((link_type, linker_uuid, linked_uuid))
        if pending is not None:
            return pending

        return self.backend.find_link(link_type, linker_uuid, linked_uuid)

    def get_link(self, linker: Entity, linked: Entity,
                 link_type: str) -> Union[LinkEntity, None]:
        """An existing link between two entities, or None."""
        spec = self.backend.get_link_type(link_type)
        if spec is None:
            return None

        linker_uuid, linked_uuid = links.canonical_pair(spec, linker.uuid, linked.uuid)
        found = self._existing_link_uuid(link_type, linker_uuid, linked_uuid)

        if found is None and not spec.symmetric and spec.endpoints_differ:
            # Tolerate the arguments arriving the other way round, as link() does
            found = self._existing_link_uuid(link_type, linked_uuid, linker_uuid)

        if found is None:
            return None

        return self._link_entity(found)

    def get_links_for(self, entity: Entity, link_type: str = None,
                      direction: str = 'both') -> list[LinkEntity]:
        """Every link touching an entity."""
        rows = self.backend.get_links_for_entity(entity.uuid, link_type=link_type,
                                                 direction=direction)

        return [self._link_entity(row['link_uuid']) for row in rows]

    def links(self, link_type: str, *_string_expressions: str,
              **_equalities) -> LinkCollection:
        """A queryable collection of links of one kind.

            ent.links('mean_response')
            ent.links('mean_response', 'mean_dff > 0.3')
            ent.links('mean_response',
                      '@Phase.index == 3 AND @Roi.has_receptive_field == True')
            ent.links('mean_response', '@linker.[Recording]imaging_rate > 8.0')

        A bare name is an attribute of the link itself; `@` addresses one of its
        endpoints, either by entity type or by role (`@linker`, `@linked`,
        `@either`, `@both`). `@linker` and `@linked` are refused for a symmetric
        kind, where which end is which is an artifact of uuid ordering.

        Everything Collection offers works here, including map_async and
        to_asdf, since a link is an entity.
        """
        spec = self.require_link_type(link_type)
        as_tree = self._build_filter_tree(_string_expressions, _equalities)

        return LinkCollection(self, spec.name, as_tree)

    def require_link_type(self, link_type: str) -> links.LinkTypeSpec:
        """The registered kind, or a message naming the ones that do exist."""
        spec = self.backend.get_link_type(link_type)
        if spec is None:
            defined = ', '.join(other.name for other in self.link_types()) or 'none'
            raise links.LinkTypeError(
                f'Link type "{link_type}" is not defined. Defined kinds: {defined}.')

        return spec

    def get_links(self, link_type: str) -> LinkCollection:
        """Deprecated alias for links()."""
        return self.links(link_type)

    def get_link_type(self, name: str) -> Union[links.LinkTypeSpec, None]:
        """The registered kind, or None if it has never been defined."""
        return self.backend.get_link_type(name)

    def link_types(self) -> list[links.LinkTypeSpec]:
        """Every registered link kind.

        Kinds are invented at runtime, so this is the only schema there is.
        """
        return self.backend.get_link_types()

    def link_from_frame(self, df, link_type: str,
                        confirm_count: int = None,
                        dry_run: bool = False) -> links.LinkWriteResult:
        """Create many links at once from a DataFrame.

        The frame needs `linker_uuid` and `linked_uuid` columns; every other
        column becomes an attribute of the link.

            ent.link_from_frame(pd.DataFrame({
                'linker_uuid': phase_uuids,
                'linked_uuid': roi_uuids,
                'mean_dff': values,
            }), 'mean_response')

        Pairs that already carry this kind are left alone, so re-running is
        harmless. Refuses a write that is a large fraction of every possible pair
        or simply enormous; see `check_write_size` for what clears each guard.

        Returns:
            links.LinkWriteResult: what was written, or would have been.
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise TypeError('link_from_frame needs a DataFrame.')

        for column in ('linker_uuid', 'linked_uuid'):
            if column not in df.columns:
                raise ValueError(f'link_from_frame needs a "{column}" column; '
                                 f'got {list(df.columns)}.')

        result = links.LinkWriteResult(link_type=link_type, requested=len(df),
                                       dry_run=dry_run)
        if len(df) == 0:
            return result

        linker_uuids = [str(value) for value in df['linker_uuid']]
        linked_uuids = [str(value) for value in df['linked_uuid']]

        # What the endpoints actually are, so the kind can be registered or checked
        kinds = self.backend.get_entity_kinds(linker_uuids + linked_uuids)
        missing = [uuid for uuid in set(linker_uuids + linked_uuids) if uuid not in kinds]
        if missing:
            raise ValueError(f'{len(missing)} endpoint uuid(s) are not entities of this '
                             f'entarchy, e.g. {missing[0]}.')

        spec = self._link_spec_for_write(link_type, kinds[linker_uuids[0]],
                                         kinds[linked_uuids[0]])

        # Validate each distinct combination once rather than once per row
        swap_by_pair = {}
        for linker_kind, linked_kind in {(kinds[a], kinds[b])
                                         for a, b in zip(linker_uuids, linked_uuids)}:
            swap_by_pair[(linker_kind, linked_kind)] = links.orientation(
                spec, linker_kind, linked_kind) == 'swapped'

        pairs = []
        for linker_uuid, linked_uuid in zip(linker_uuids, linked_uuids):
            if swap_by_pair[(kinds[linker_uuid], kinds[linked_uuid])]:
                linker_uuid, linked_uuid = linked_uuid, linker_uuid
            pairs.append(links.canonical_pair(spec, linker_uuid, linked_uuid))

        # Drop duplicates within the input, keeping the first occurrence
        seen = {}
        for index, pair in enumerate(pairs):
            seen.setdefault(pair, index)
        keep_indices = sorted(seen.values())
        result.duplicates_dropped = len(pairs) - len(keep_indices)

        unique_pairs = [pairs[index] for index in keep_indices]

        already = self.backend.find_existing_pairs(spec.name, unique_pairs)
        already |= {pair for pair in unique_pairs
                    if (spec.name, pair[0], pair[1]) in self._pending_link_index}
        fresh = [(index, pair) for index, pair in zip(keep_indices, unique_pairs)
                 if pair not in already]
        result.already_present = len(unique_pairs) - len(fresh)

        links.check_write_size(spec, len(fresh),
                               len({pair[0] for pair in unique_pairs}),
                               len({pair[1] for pair in unique_pairs}),
                               confirm_count=confirm_count)

        if spec.cardinality == 'one_per_linker':
            self._check_one_per_linker(spec, [pair for _, pair in fresh])

        result.created = len(fresh)
        if dry_run or len(fresh) == 0:
            return result

        self._write_links(spec, fresh, df, result)

        return result

    def _check_one_per_linker(self, spec, pairs) -> None:
        wanted = [linker for linker, _ in pairs]
        present = self.backend.count_links_per_linker(spec.name, wanted)

        counts = {}
        for linker in wanted:
            counts[linker] = counts.get(linker, 0) + present.get(linker, 0) + 1

        offenders = [linker for linker, count in counts.items() if count > 1]
        if offenders:
            raise links.LinkCardinalityError(
                f'"{spec.name}" is declared one_per_linker, but this write would leave '
                f'{len(offenders)} linker(s) with more than one, e.g. {offenders[0]}.')

    def _write_links(self, spec, fresh, df, result) -> None:
        """Insert carrier entities, link rows and attributes, in that order."""
        import pandas as pd

        link_uuids = [str(uuid.uuid4()) for _ in fresh]
        result.link_uuids = link_uuids

        entity_records = [
            {'uuid': link_uuid, 'id': f'{spec.name}@{pair[1]}',
             'parent_uuid': pair[0], 'entity_type_name': 'LinkEntity'}
            for link_uuid, (_, pair) in zip(link_uuids, fresh)]

        link_records = [
            {'link_uuid': link_uuid, 'link_type': spec.name,
             'linker_uuid': pair[0], 'linked_uuid': pair[1]}
            for link_uuid, (_, pair) in zip(link_uuids, fresh)]

        attribute_columns = [column for column in df.columns
                             if column not in ('linker_uuid', 'linked_uuid')]

        with self.backend.batch():
            self.backend.add_entity_records(entity_records)
            self.backend.add_links(link_records)

            if attribute_columns:
                source_rows = [index for index, _ in fresh]
                attributes = pd.DataFrame(
                    {column: df[column].iloc[source_rows].to_numpy()
                     for column in attribute_columns},
                    index=link_uuids)
                self.backend.set_attributes_by_uuid(self, attributes)

    def link_from_matrix(self, linkers, linkeds, matrix, link_type: str,
                         where, value_name: str = 'value',
                         confirm_count: int = None,
                         dry_run: bool = False) -> links.LinkWriteResult:
        """Create links from a pairwise matrix, keeping only what `where` selects.

            r = np.corrcoef(traces)
            ent.link_from_matrix(rois, rois, r, 'correlated',
                                 where=lambda v: abs(v) > 0.6, value_name='r')

        `where` is required rather than optional, which is the point of this
        method: forgetting to threshold is how a pairwise result becomes millions
        of rows, and a signature that cannot be called without one removes the
        mistake instead of catching it afterwards.

        For a symmetric kind over the same entities only the upper triangle is
        considered, since each pair means one link.
        """
        import numpy as np
        import pandas as pd

        if where is None:
            raise ValueError(
                'link_from_matrix needs a "where" predicate or threshold. Storing every '
                'pair is what makes a pairwise result unmanageable; if that really is '
                'what you want, build the frame yourself and use link_from_frame.')

        matrix = np.asarray(matrix)
        linker_uuids = [item if isinstance(item, str) else item.uuid for item in linkers]
        linked_uuids = [item if isinstance(item, str) else item.uuid for item in linkeds]

        if matrix.shape != (len(linker_uuids), len(linked_uuids)):
            raise ValueError(f'matrix is {matrix.shape} but there are '
                             f'{len(linker_uuids)} linkers and {len(linked_uuids)} linked '
                             f'entities.')

        keep = where(matrix) if _accepts_array(where, matrix) else np.vectorize(where)(matrix)
        keep = np.asarray(keep, dtype=bool)

        spec = self.backend.get_link_type(link_type)
        same_entities = linker_uuids == linked_uuids
        if same_entities and (spec is None or spec.symmetric):
            # Each unordered pair means one link, and the diagonal is not a pair
            keep &= np.triu(np.ones_like(keep, dtype=bool), k=1)
        elif same_entities:
            np.fill_diagonal(keep, False)

        rows, columns = np.nonzero(keep)

        frame = pd.DataFrame({
            'linker_uuid': [linker_uuids[row] for row in rows],
            'linked_uuid': [linked_uuids[column] for column in columns],
            value_name: matrix[rows, columns],
        })

        return self.link_from_frame(frame, link_type, confirm_count=confirm_count,
                                    dry_run=dry_run)

    def matrix_from_links(self, linkers: Collection, linkeds: Collection,
                          link_type: str, value_name: str,
                          *_string_expressions: str, **_equalities) -> 'pd.DataFrame':
        """The links between two collections, as a matrix in their own order.

            r = ent.matrix_from_links(day1_rois, day2_rois, 'same_cell', 'overlap')
            r = ent.matrix_from_links(rois, rois, 'correlation', 'r')
            r = ent.matrix_from_links(rois, rois, 'correlation', 'r', 'r > 0.5')

        The counterpart to `link_from_matrix`: rows are `linkers` in their own
        order, columns are `linkeds` in theirs, and a pair with no link - or one
        the filter expressions exclude - is NaN. `rois.matrix_from_links(...)`
        is the same thing said from the collection.

        Two queries however large the result: the ends of every link at once,
        rather than one lookup per entity, and the values through the same
        pivot `dataframe_of` uses. A link's ends are in the links table and its
        values in the attributes table, which is why it takes two.

        For a symmetric kind each link stands at both of its positions, since
        which end is stored as the linker is an artifact of uuid ordering
        there. So `matrix_from_links(rois, rois, ...)` comes back symmetric
        rather than triangular, and two disjoint collections come back the way
        they were asked for either way.

        Args:
            linkers: the collection the rows are.
            linkeds: the collection the columns are.
            link_type: the kind of link to read.
            value_name: which attribute of the link fills the cells.
            *_string_expressions: filters on the links, as `links()` takes them.
            **_equalities: the same, as equalities.

        Returns:
            pandas.DataFrame: indexed by the uuids of `linkers`, with the uuids
            of `linkeds` as columns.
        """
        import numpy as np
        import pandas as pd

        from .links import orientation

        pairs = linkers.links_to(linkeds, link_type,
                                 *_string_expressions, **_equalities)
        spec = pairs.link_type_spec

        row_uuids = [_uuid for _uuid, _ in linkers._rows()]
        column_uuids = [_uuid for _uuid, _ in linkeds._rows()]

        endpoints = self.backend.get_link_endpoints(pairs)

        if len(endpoints) == 0:
            matrix = pd.DataFrame(np.nan, index=row_uuids, columns=column_uuids)
        else:
            # links_to() stores the pair the way the kind's endpoint types say,
            #  which for a kind running the other way is not the way it was
            #  asked for here. Naming the two columns accordingly is the whole
            #  of putting the matrix back the right way round.
            swapped = orientation(spec, linkers._as_endpoint(),
                                  linkeds._as_endpoint()) == 'swapped'
            names = (['link_uuid', 'column_uuid', 'row_uuid'] if swapped
                     else ['link_uuid', 'row_uuid', 'column_uuid'])

            frame = pd.DataFrame(endpoints, columns=names)

            if spec.symmetric:
                flipped = frame.rename(columns={'row_uuid': 'column_uuid',
                                                'column_uuid': 'row_uuid'})
                frame = pd.concat([frame, flipped], ignore_index=True)
                # A link from an entity to itself lands on one cell twice
                frame = frame.drop_duplicates(
                    subset=['link_uuid', 'row_uuid', 'column_uuid'])

            frame = frame.join(pairs.dataframe_of([value_name]), on='link_uuid')

            try:
                matrix = frame.pivot(index='row_uuid', columns='column_uuid',
                                     values=value_name)
            except ValueError as exc:
                raise ValueError(
                    f'More than one "{link_type}" link falls on the same pair, so '
                    f'there is no single {value_name} to put in that cell.') from exc

            matrix = matrix.reindex(index=row_uuids, columns=column_uuids)

        try:
            matrix = matrix.astype(float)
        except (TypeError, ValueError):
            # A link may carry a string or a date as well as a number, and those
            #  are left as they are rather than refused
            pass

        matrix.index.name = linkers.entity_type.__name__
        matrix.columns.name = linkeds.entity_type.__name__

        return matrix

    def redefine_link_type(self, name: str, *args, delete_existing: bool = False,
                           **kwargs) -> links.LinkTypeSpec:
        """Replace a link kind's definition.

        Refuses while links of that kind exist, since they were written against
        the old constraints and may not satisfy the new ones. Pass
        delete_existing=True to drop them first.
        """
        existing_count = self.backend.count_links_of_type(name)
        if existing_count > 0 and not delete_existing:
            raise links.LinkTypeError(
                f'{existing_count} link(s) of type "{name}" exist and were written '
                f'against its current definition. Pass delete_existing=True to remove '
                f'them and redefine it.')

        if existing_count > 0:
            self.backend.remove_links_of_type(name)

        self.backend.remove_link_type(name)

        return self.define_link_type(name, *args, **kwargs)

    def to_asdf(self, destination: str, **kwargs) -> dict:
        """Write this entarchy to a self-describing ASDF archive.

        The archive is itself an entarchy - `Entarchy(destination)` opens it -
        so analysis and figure code reads it without modification. It is
        read-only; `entarchy.tools.archive import` turns one back into a
        writable entarchy.

        Returns:
            dict: counts of what was written, including a portability report.
        """
        from ..tools import archive

        return archive.export(self, destination, **kwargs)

    def forget_entity(self, entity: Union[Entity, str]) -> None:
        """Drop an entity from the in-memory registry.

        Used by bulk operations to keep the registry from growing with every
        processed entity. The entity object stays usable and is reloaded on
        demand the next time it is looked up by UUID.

        Args:
            entity (Entity or str): The entity to release, or its UUID.

        Returns:
            None
        """

        _uuid = entity if isinstance(entity, str) else entity.uuid

        if _uuid in self._entities_to_update:
            self._entities_to_update.remove(_uuid)

        self._entities.pop(_uuid, None)

    def _remove_entity_from_update(self, entity: Entity) -> None:
        """Unmark an entity for update in the backend.

        Args:
            entity (Entity): The entity to unmark for update.

        Returns:
            None
        """

        if entity.uuid in self._entities_to_update:
            self._entities_to_update.remove(entity.uuid)

    def set_current_analysis(self, _analysis: Union[AnalysisEntity, str]) -> None:
        """Set the current analysis for the entarchy system.

        Args:
            _analysis (Union[AnalysisEntity, str]): The analysis to set as current, or its UUID.

        Returns:
            None
        """

        if isinstance(_analysis, str):
            res = self.backend.get_entities_of_type(AnalysisEntity.__name__)

            # Check string against id
            existing_data = [r for r in res if r[1] == _analysis]

            if len(existing_data) == 0:
                _analysis_entity = AnalysisEntity(_analysis, _entarchy=self)
                self._add_entity(_analysis_entity)
                self.commit()
            else:
                # Reuse the persisted UUID. Creating a fresh entity object here would
                #  generate a new random UUID each session, and attributes written on
                #  the analysis in an earlier session would no longer be reachable.
                _analysis_entity = AnalysisEntity(_analysis, _entarchy=self, _uuid=existing_data[0][0])
        elif isinstance(_analysis, AnalysisEntity):
            _analysis_entity = _analysis
        else:
            raise ValueError(f'Invalid analysis type {type(_analysis)}. Must be of type Analysis or str.')

        self._current_analysis = _analysis_entity

    def start_digest(self):
        self._is_in_digest_mode = True

    def end_digest(self):
        self._is_in_digest_mode = False


def digest_method(fun: Callable):

    def _digest_fun(ent: Entarchy, *args, **kwargs):
        ent.start_digest()
        try:
            result = fun(ent, *args, **kwargs)
        finally:
            ent.end_digest()
        return result

    return _digest_fun


def _accepts_array(predicate, sample) -> bool:
    """Whether a predicate handles a whole array rather than one value at a time."""
    try:
        result = predicate(sample)
    except Exception:
        return False

    return getattr(result, 'shape', None) == sample.shape
