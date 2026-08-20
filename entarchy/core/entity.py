from __future__ import annotations

import atexit
import datetime
import functools
import html
import re
import sys
import time
import traceback
import uuid
from typing import (Any, Callable, Generator, Generic, Type, TypeVar,
                    TYPE_CHECKING, Union, overload)

import alive_progress
import numpy as np
import pandas as pd

from . import console
from . import query
from .links import LinkTypeError

if TYPE_CHECKING:
    from . import describe as describe_mod
    from .entarchy import Entarchy
    from ..backend.blob_store import MediaFile

# What sort of entity a collection holds. Bound to Entity so a collection
#  can only ever be of entities, and carried through get(), indexing and
#  iteration so that ent.get(Roi)[0] is a Roi to a type checker rather
#  than the base class - which is what makes a schema's own properties and
#  methods complete in an editor.
EntityT = TypeVar('EntityT', bound='Entity')


class Entity(object):
    """Base class for all entities in the Entarchy system.
    This class can be extended to create specific types of entities.
    """
    # Setup attributes
    _child_entity_types: list[Type[Entity]] = None
    collection_type: Type[Collection] = None

    # Runtime attributes
    _is_in_context: bool = False

    def __init__(self,
                 _entarchy: Entarchy,
                 _uuid: str | None = None,
                 _id: str | None = None,
                 _parent: Entity | None = None,
                 _init_cache: dict[str, Any] | None = None):

        # __new__ may have returned an already-registered instance (identity map).
        #  In that case, __init__ runs again on that instance and must NOT reset its
        #  state - doing so would wipe the attribute cache and silently discard any
        #  pending, uncommitted attribute updates. Only merge in new cache values.
        if getattr(self, '_entity_initialized', False):
            if _init_cache is not None:
                if not isinstance(_init_cache, dict):
                    raise TypeError('_init_cache must be a dictionary of attribute names and values.')
                for k, v in _init_cache.items():
                    self._attribute_cache.setdefault(k, v)
            return

        self._entarchy = _entarchy
        self._uuid = _uuid
        self._id = _id
        self._parent = _parent

        if self._uuid is None and self._id is None:
            raise ValueError("Need to provide either _id or _uuid")

        # Create PK if not provided
        if self._uuid is None:
            self._uuid = uuid.uuid4()

        # Set up attribute cache
        self._attribute_cache_start_time = datetime.datetime.now()
        self._attribute_cache: dict[str, Any] = {}
        self._attributes_to_update: list[str] = []

        # Add entity to entarchy object
        self.entarchy.add_existing_entity(self)

        # Initialize cache if provided
        if _init_cache is not None:
            if not isinstance(_init_cache, dict):
                raise TypeError('_init_cache must be a dictionary of attribute names and values.')
            self._attribute_cache.update(_init_cache)

        self._entity_initialized = True

    @overload
    def __new__(cls: Type[EntityT], _expression: str, /,
                *args, **kwargs) -> DeferredEntityCollection[EntityT]: ...

    @overload
    def __new__(cls: Type[EntityT], _entarchy: Entarchy = ...,
                _uuid: str | None = ..., _id: str | None = ...,
                _parent: Entity | None = ...,
                _init_cache: dict[str, Any] | None = ...) -> EntityT: ...

    def __new__(cls, *args, **kwargs):
        # Two constructions wearing one name. Given a string this builds a
        #  deferred collection rather than an entity - Roi('index > 3') - and
        #  Python then skips __init__ because what came back is not a Roi. The
        #  overloads say so; without them a type checker reads the expression
        #  form as a Roi and refuses every method the collection has.

        # _entarchy = args[0] if len(args) > 0 else kwargs.get('_entarchy', None)
        arg0 = args[0] if len(args) > 0 else None

        if isinstance(arg0, str):
            obj = DeferredEntityCollection(cls, arg0)
            return obj

        _entarchy = arg0 if arg0 is not None else kwargs.get('_entarchy', None)
        if _entarchy is None:
            # Pickle reconstruction calls __new__ without constructor args.
            # In that case, create a blank instance and let pickle restore state.
            #  (Only applies during multiprocessing map calls, where the entity is reconstructed in a new process)
            if len(args) == 0 and len(kwargs) == 0:
                return super().__new__(cls)

            raise ValueError('_entarchy argument is required')

        _uuid = args[1] if len(args) > 1 else kwargs.get('_uuid', None)
        if _uuid is not None and _uuid in _entarchy:
            obj = _entarchy.get_entity_by_uuid(_uuid)

            # Update cache if provided
            # _init_cache = kwargs.get('_init_cache', None)
            # if _init_cache is not None:
            #     obj.update_cache(_init_cache)

            return obj

        return super().__new__(cls)

    def __contains__(self, item: str) -> bool:
        """Check if the entity has a dynamic attribute.

        Dynamic attribute keys must always be strings.
        Using a list or tuple of strings will check if all attributes exist.

        Args:
            item (str): The key for the attribute to check.

        Raises:
            TypeError: If item is not a string.

        Returns:
            bool: True if the attribute(s) exist, False otherwise.
        """

        if not isinstance(item, str):
            raise TypeError('Item must be a string')

        # Cached attributes exist by definition; saves a database query
        if item in self._attribute_cache:
            return True

        return self.entarchy.backend.has_entity_attribute(self, item)

    def __enter__(self):
        # Set context flag
        self._is_in_context = True

        # Set current analysis if applicable
        if isinstance(self, AnalysisEntity):
            setattr(self, '__prev_analysis', self.entarchy.current_analysis)
            self.entarchy.set_current_analysis(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Commit any pending changes, unless the context is left with an exception
        #  (in that case partially processed state must not be persisted)
        if exc_type is None:
            self.commit()

        # Reset context flag
        self._is_in_context = False

        # Reset current analysis if applicable
        if hasattr(self, '__prev_analysis'):
            self.entarchy.set_current_analysis(getattr(self, '__prev_analysis'))
            delattr(self, '__prev_analysis')

    def __hash__(self):
        """Use integer representation of unique identifier as hash"""
        return uuid.UUID(self.uuid).int

    def __getitem__(self, key: Union[str, list[str]]) -> Union[Any, tuple[Any, ...]]:
        """Get a dynamic attribute from the entity.

        Dynamic attribute keys must always be strings.
        Using a list or tuple of strings will return a list or tuple of the same length.

        Args:
            key (str or list of str): The key(s) for the attribute(s) to get.

        Raises:
            TypeError: If key is not a string or list/tuple of strings.

        Returns:
            Any or tuple of Any: The value(s) of the requested attribute(s).
        """

        # Validate key type
        if not isinstance(key, (str, list)):
            raise TypeError('Key must be a string or list of strings.')

        if isinstance(key, list):
            if not all(isinstance(k, str) for k in key):
                raise TypeError('List of keys must contain only strings.')
        else:
            key = [key]

        # TODO: As expected this causes massive performance issues. Nice idea. Optimize later.
        #  Better idea: use a time-to-live cache with fixed (configurable) expiration time
        # # Reset attribute cache if any key is in cache and entity is dirty
        # if any([k in self._attribute_cache for k in key]):
        #     modified = self.entarchy.backend.get_entity_modified_time(self)
        #
        #     if modified > self._attribute_cache_start_time:
        #         self._attribute_cache_start_time = datetime.datetime.now()
        #         self._attribute_cache = {}

        # Load missing attributes from backend
        keys_to_load = list(set(key) - set(self._attribute_cache.keys()))
        if len(keys_to_load) > 0:
            if len(keys_to_load) == 1:
                values = [
                    self.entarchy.backend.get_entity_attribute(self, keys_to_load[0])]
            else:
                values = self.entarchy.backend.get_entity_attributes(self, keys_to_load)

            # Update cache
            for k, v in zip(keys_to_load, values):
                self._attribute_cache[k] = v

        # Retrieve in order
        res = tuple(self._attribute_cache[k] for k in key)

        # Return
        if len(res) > 1:
            return res
        return res[0]

    def links(self, link_type: str = None, direction: str = 'both') -> list[LinkEntity]:
        """The links touching this entity.

        Args:
            link_type: restrict to one kind, or None for all of them.
            direction: 'out' for links where this entity is the linker, 'in' for
                those where it is the linked end, 'both' by default. For a
                symmetric kind the two are an artifact of uuid ordering, so only
                'both' is meaningful there.
        """
        return self.entarchy.get_links_for(self, link_type=link_type, direction=direction)

    def link_types(self) -> list[str]:
        """Which kinds of link touch this entity."""
        return list(self.entarchy.backend.count_links_by_type(self.uuid))

    def link_counts(self) -> dict[str, int]:
        """How many links of each kind touch this entity, keyed by kind."""
        return self.entarchy.backend.count_links_by_type(self.uuid)

    def set_media(self, name: str, source, media_type: str = None,
                  move: bool = False) -> 'MediaFile':
        """Take a file into the entarchy and attach it under `name`.

        For anything large and opaque - video, raw image stacks, a protocol
        document. The file is copied under the entarchy's `media/` directory, so
        the entarchy stays self-contained and can be moved or copied whole:

            recording.set_media('video', '/data/behaviour.avi')
            recording['video'] = MediaFile('/data/behaviour.avi')   # equivalent

        A method as well as an assignment because copying a gigabyte should not
        look like setting a value. `move=True` takes the source instead of
        copying it, for an ingest that owns the file.

        Reading the attribute back gives a MediaFile, which is os.PathLike - so
        it goes straight to whatever reads that kind of file. entarchy never
        decodes it.

        Returns:
            MediaFile: the stored file, as reading the attribute would give it.
        """
        from ..backend.blob_store import MediaFile

        self[name] = (source if isinstance(source, MediaFile)
                      else MediaFile(source, media_type=media_type, move=move))

        return self[name]

    def describe(self, values: bool = True, verify: bool = False) -> 'describe_mod.Description':
        """What this entity holds: attributes, links, media, children, ancestry.

            roi.describe()                  # renders whole
            roi.describe().attributes       # a DataFrame

        Blobs are reported by type and size rather than read, so this costs a
        handful of indexed queries however much data the entity carries. The
        sizes are bytes as stored - the encoded, compressed container - not the
        size in memory once decoded.

        Args:
            values: read and inline the values of scalar attributes. They are
                cheap and are their own summary; pass False for the shape alone.
            verify: re-hash media files to check they are intact. Off by
                default because it re-reads every byte of them.

        Returns:
            Description: sections that are DataFrames, rendering together.
        """
        from . import describe as describe_mod

        backend = self.entarchy.backend
        notes = []

        metadata = backend.get_entity_attribute_metadata(self.uuid)

        inlined = {}
        if values:
            scalar_names = [name for name, data_type, _ in metadata
                            if data_type in describe_mod.SCALAR_TYPES]
            if len(scalar_names) > 0:
                try:
                    read = self[scalar_names]
                    if len(scalar_names) == 1:
                        read = (read,)
                    inlined = dict(zip(scalar_names, read))
                except Exception as exc:
                    notes.append(f'scalar values could not be read: {exc}')

        try:
            counts = self.link_counts()
        except Exception:
            counts = {}

        carried = {}
        if counts:
            total_links = sum(counts.values())
            if total_links <= describe_mod.LINK_NAME_LIMIT:
                try:
                    carried = backend.get_link_attribute_names(self.uuid)
                except Exception as exc:
                    notes.append(f'link attribute names could not be read: {exc}')
            else:
                notes.append(
                    f'{total_links} links: what they carry was not looked up, '
                    f'which needs a scan of all of them past '
                    f'{describe_mod.LINK_NAME_LIMIT}.')

        try:
            children = backend.count_child_entities(self.uuid)
        except Exception:
            children = {}

        # A link's carrier entity is parented to its linker, which keeps the
        #  entity tree valid - but they are links, they have their own section,
        #  and counting them here would tell a ROI it has 20 000 children
        children.pop(describe_mod.LINK_ENTITY_TYPE, None)

        headline = {'type': type(self).__name__, 'id': self.id, 'uuid': self.uuid}
        try:
            headline['path'] = self.path
        except Exception:
            pass

        media_names = []
        try:
            media_names = self.media()
        except Exception:
            pass

        sections = {
            'attributes': describe_mod.attribute_rows(metadata, inlined, media_names),
            'links': describe_mod.link_rows(counts, carried),
            'media': describe_mod.media_rows(self, media_names, verify=verify),
            'children': describe_mod.count_rows(children, 'type', 'count'),
            'ancestry': self._ancestry_rows(),
        }

        return describe_mod.Description(
            subject=f'{type(self).__name__} {self.id}',
            headline=headline, sections=sections, notes=notes)

    def _ancestry_rows(self) -> 'pd.DataFrame':
        """This entity's parents, outermost first.

        The entarchy root is left out: every entity has it, so a row saying so
        carries nothing. An entity at the top of the hierarchy therefore has no
        ancestry section rather than one row of nothing.

        `seen` guards the walk rather than trusting the tree, because a
        description is reached for when something is already wrong and a cycle
        must not hang the session.
        """
        chain = []
        entity = self.parent
        seen = set()

        while entity is not None and entity.uuid not in seen:
            seen.add(entity.uuid)
            if not isinstance(entity, EntarchyEntity):
                chain.append({'type': type(entity).__name__, 'id': entity.id})
            entity = entity.parent

        return pd.DataFrame(list(reversed(chain)), columns=['type', 'id'])

    def media(self) -> list[str]:
        """Which attributes of this entity are media files.

        Reads pointers only, so it costs nothing even when the entity also
        holds large blobs - asking `self[name]` for each would decode every one
        of them.
        """
        return self.entarchy.backend.get_media_attribute_names(self.uuid)

    def __repr__(self):
        return f'{self.__class__.__name__}(id=\'{self.id}\' uuid=\'{self.uuid}\')'

    def _repr_html_(self) -> str:
        """Rich representation for notebooks.

        Lists attribute names but deliberately does not read their values: a single
        entity can hold hundreds of megabytes of array data, which must not be
        loaded just because it was the last expression in a cell.

        Link kinds are listed alongside the attributes, since links are the half
        of an entity that keys() does not show. The line is omitted when nothing
        links to this entity, so an entarchy that uses no links reads as before.
        """
        try:
            rows = [('type', self.__class__.__name__),
                    ('id', self.id),
                    ('uuid', self.uuid)]

            try:
                rows.append(('path', self.path))
            except Exception:
                pass

            names = sorted(self.keys())
            shown = names[:_HTML_MAX_ATTRIBUTES]
            listing = ', '.join(html.escape(str(n)) for n in shown)
            if len(names) > len(shown):
                listing += f' <span style="color:#888">... and {len(names) - len(shown)} more</span>'

            try:
                counts = self.link_counts()
            except Exception:
                # A backend without links, or one that cannot be reached
                counts = {}

            body = ''.join(
                f'<tr><td style="text-align:right;color:#888;padding-right:8px">{html.escape(str(k))}</td>'
                f'<td style="font-family:monospace">{html.escape(str(v))}</td></tr>'
                for k, v in rows)

            return (f'<div>'
                    f'<table style="border:none">{body}'
                    f'<tr><td style="text-align:right;color:#888;padding-right:8px;vertical-align:top">'
                    f'{len(names)} attributes</td><td>{listing}</td></tr>'
                    f'{_links_row_html(counts)}'
                    f'</table>'
                    f'<div style="color:#888;font-size:90%">read values with '
                    f'<code>entity[&#39;name&#39;]</code>'
                    f'{", links with <code>entity.links(&#39;kind&#39;)</code>" if counts else ""}'
                    f'</div>'
                    f'</div>')
        except Exception:
            # A repr must never break a notebook session
            return _fallback_html(self)

    def __setitem__(self, key: Union[str, list[str]], value: Any):
        """Set a dynamic attribute on the entity.

        Dynamic attribute keys must always be strings.
        Using a list of strings it's expected that the values to be a list or tuple of the same length.

        Args:
            key (str or list of str): The key(s) for the attribute(s) to set.
            value (Any or list/tuple of Any): The value(s) for the attribute(s) to set.

        Raises:
            TypeError: If key is not a string or list of strings.
            TypeError: If value is not a list when key is a list/tuple.
            ValueError: If lengths of key and value lists do not match.

        """

        if not isinstance(key, (str, list, tuple)):
            raise TypeError('Key must be a string or list or tuple of strings.')

        if isinstance(key, (list, tuple)):
            if not all(isinstance(k, str) for k in key):
                raise TypeError('List or tuple of keys must contain only strings.')

            if not isinstance(value, (list, tuple)):
                raise TypeError('Value must be a list or tuple when key is a list or tuple.')

            if len(key) != len(value):
                raise ValueError('Length of key list/tuple must match length of value list/tuple.')

        else:
            key = [key]
            value = [value]

        # Reject identity rewrites up front, before any cache state is touched
        #  (they are only written once, while the entity is queued for insertion)
        for k in key:
            if k in ('id', 'uuid') and not self.entarchy.is_pending_add(self):
                raise RuntimeError(f'Attribute "{k}" is the entity identity '
                                   f'and cannot be modified on {self}.')

        # Update attribute(s) in cache and mark for update
        for k, v in zip(key, value):

            self._attribute_cache[k] = v

            if k not in self._attributes_to_update:
                self._attributes_to_update.append(k)

        self.entarchy.add_entity_for_update(self)

        # If not in context, update immediately
        #  (commit() defers entities that are still queued for insertion)
        if not self.is_in_context and not self.entarchy.is_in_context:
            self.commit()

    @classmethod
    def add_child_entity_type(cls, entity_type):
        """Register a child entity type for this entity class.
        """

        if isinstance(entity_type, Entity):
            raise TypeError('entity_type must be a class, not an instance')

        if not issubclass(entity_type, Entity):
            raise TypeError('entity_type must be a subclass of Entity')

        if cls._child_entity_types is None:
            cls._child_entity_types = []
        cls._child_entity_types.append(entity_type)

    @classmethod
    def get_child_entity_types(cls):
        # return [globals()[c] if isinstance(c, str) else c for c in cls._child_entity_types] Does not work yet for str
        return cls._child_entity_types

    @classmethod
    def get_collection_type(cls) -> type[Collection]:
        if cls.collection_type is None:
            return Collection
        return cls.collection_type

    @property
    def entarchy(self) -> Entarchy:
        return self._entarchy

    @property
    def id(self) -> str:
        """Get the ID of the entity.

        Returns:
            str: The ID of the entity.
        """
        return self._id

    @property
    def is_in_context(self) -> bool:
        """Check if the entity is in a context manager.

        Returns:
            bool: True if the entity is in a context manager, False otherwise.
        """
        return self._is_in_context

    @property
    def uuid(self) -> str:
        """Get the UUID primary key of the entity.

        Returns:
            str: The UUID string representation of the entity's primary UUID key
        """
        return str(self._uuid)

    @property
    def parent(self) -> Union[Entity, None]:
        """Get the parent entity of this entity.

        Returns:
            Entity: The parent entity or None.
        """
        if self._parent is None:
            res = self.entarchy.backend.get_entity_parent(self)
            if res is None:
                self._parent = False
            else:
                self._parent = self.entarchy.get_entity(entity_type_name=res[0], _uuid=res[1], _id=res[2])

        return None if not self._parent else self._parent

    @property
    def path(self) -> str:
        parent = self.parent
        if parent is not None and not isinstance(parent, EntarchyEntity):
            return f'{parent.path}/{self.id}'
        return self.id

    def commit(self):

        # Attributes cannot be written before the entity row exists. Entities that
        #  are still queued for insertion stay on the update list; Entarchy.commit()
        #  inserts them first and commits their attributes right after.
        if self.entarchy.is_pending_add(self):
            return

        # Remove entity from entarchy update list
        self.entarchy.remove_entity_from_update(self)

        # Update attributes in backend if updates are pending
        if len(self._attributes_to_update) > 0:

            names = self._attributes_to_update
            values = [self._attribute_cache[n] for n in names]
            # print(f'Update entity attributes: {names}')
            # print(f'Values: {values}')

            if len(names) > 1:
                res = self.entarchy.backend.set_entity_attributes(self, names, values)
            else:
                res = self.entarchy.backend.set_entity_attribute(self, names[0], values[0])

            if not res:
                raise RuntimeError(f'Failed to update entity attributes {names} in backend.')

            # A MediaFile assigned to an attribute names a source file outside
            #  the entarchy; what was stored is the copy taken of it. Keeping the
            #  source in the cache would hand back a handle to a file entarchy
            #  does not own and may not keep, so the next read comes from the row.
            from ..backend.blob_store import MediaFile

            for name, written in zip(names, values):
                if isinstance(written, MediaFile) and not written.is_stored:
                    self._attribute_cache.pop(name, None)

            # Reset list
            self._attributes_to_update = []

        # In digest mode, purge cache from memory after commit
        if self.entarchy.is_in_digest_mode:
            self._attribute_cache = {}

    def keys(self):
        """Return a list of all dynamic attribute keys for this entity.
        """
        return self.entarchy.backend.get_entity_attribute_names(self)

    def _ipython_key_completions_(self) -> list[str]:
        """The stored names, for tab completion inside the brackets.

        IPython asks an object for this when the cursor is inside a subscript.
        Nothing here defined it, so there was nothing entarchy-specific to
        offer and the completer fell back to the global namespace: tabbing
        inside an entity's brackets suggested 335 builtins and magics and not
        one of the names actually stored.

        They cannot be reached any other way. `s2p/npix` is a row in the
        attributes table rather than a Python attribute, so `dir()` cannot see
        it and no static tool can know it - but one indexed query can.
        """
        try:
            return self.keys()
        except Exception:
            # This runs on a keystroke. An entity whose backend has gone away
            #  should still be typeable, so a completer must never raise
            return []

    def to_dict(self) -> dict[str, Any]:
        """Return a dictionary of all dynamic attributes for this entity.
        """
        keys = self.keys()
        values = self[keys]
        return dict(zip(keys, values))

    def update(self, attribute_dict: dict[str, Any]):
        for k, v in attribute_dict.items():

            if not isinstance(k, str):
                raise TypeError('Attribute keys must be strings.')

            self.__setitem__(k, v)


class EntarchyEntity(Entity):
    """Root entarchy entity
    """


class AnalysisEntity(Entity):
    """A named analysis, parentless, holding whatever that analysis derived.

    `set_current_analysis(name)` creates or reuses one by id, so a name is
    the same entity - and the same uuid - in every session. It is also a
    context manager: `with analysis:` makes it current and restores the
    previous one on exit.

    Segregation is by entity, not by stamp. Two analyses that each write
    `analysis['dff']` write two rows, because the attributes primary key is
    `(entity_uuid, name)`. Writing a derived value onto the entity it was
    derived from has no such protection - the second analysis overwrites the
    first - so a name several analyses share belongs on the analysis, or in
    a namespace of its own (`cmn_rf/dff`).
    """


class LinkEntity(Entity):
    """A relationship between two entities, carrying data of its own.

    Every link is a LinkEntity regardless of its kind, because kinds are data
    rather than classes. `link_type` says which kind this one is; `linker` and
    `linked` are its endpoints.

    For a symmetric kind the two endpoints are stored in uuid order, so which is
    which carries no meaning - use `other_end(entity)` rather than assuming.
    """

    _link_row: dict = None

    @property
    def row(self) -> dict:
        if self._link_row is None:
            self._link_row = self.entarchy.backend.get_link_row(self.uuid)
            if self._link_row is None:
                raise RuntimeError(f'{self} has no link row. It may have been created '
                                   f'as a plain entity rather than through Entarchy.link.')

        return self._link_row

    @property
    def link_type(self) -> str:
        return self.row['link_type']

    @property
    def linker(self) -> Entity:
        return self.entarchy.get_entity_by_uuid(self.row['linker_uuid'])

    @property
    def linked(self) -> Entity:
        return self.entarchy.get_entity_by_uuid(self.row['linked_uuid'])

    def other_end(self, entity: Union[Entity, str]) -> Entity:
        """The endpoint that is not the one given.

        The way to traverse a symmetric link, where linker and linked are an
        artifact of uuid ordering rather than meaning.
        """
        entity_uuid = entity if isinstance(entity, str) else entity.uuid

        if entity_uuid == self.row['linker_uuid']:
            return self.linked
        if entity_uuid == self.row['linked_uuid']:
            return self.linker

        raise ValueError(f'{entity_uuid} is not an endpoint of {self}.')

    def __repr__(self):
        row = self._link_row
        if row is None:
            return f'LinkEntity(uuid=\'{self.uuid}\')'
        return (f'LinkEntity({row["link_type"]}: {row["linker_uuid"][:8]} -> '
                f'{row["linked_uuid"][:8]})')


class Collection(Generic[EntityT]):
    """Base class for collection of entities
    """

    # TODO: implement .apply method along column axis similar to pandas DataFrame.apply() ?

    _as_tree: dict[str, Any]
    _length: int = None
    _name = None

    def __init__(self,
                 _entarchy: Entarchy,
                 _entity_type: Type[EntityT],
                 _as_tree: dict[str, Any]):
        self._entity_type = _entity_type
        self._entarchy = _entarchy
        self._as_tree = _as_tree

        self._cache = pd.DataFrame()
        self._pending_changes: dict[str, list[int]] = {}
        self._init_time = datetime.datetime.now()

        # What sort() asked for, and the (uuid, id) order it works out to.
        #  Both None on an unsorted collection, which is the default and keeps
        #  the backend's uuid order.
        self._sort_keys: list[tuple[str, bool]] = None
        self._sort_natural: bool = False
        self._sort_missing: str = 'last'
        self._sort_order: list[tuple[str, str]] = None
        # Resolving the order reads attributes, which goes back through
        #  dataframe_of, which asks for the order. Set while that is in flight.
        self._sort_resolving: bool = False

    def _derive(self, _as_tree) -> Collection[EntityT]:
        """A collection like this one, with a different filter tree.

        Set operations and where() build new collections, and a subclass that
        carries extra state - LinkCollection and its kind - has to keep it.
        """
        collection_type = self.entity_type.collection_type or Collection

        return collection_type(self.entarchy, self.entity_type, _as_tree)

    def sort(self, *keys: str, natural: bool = True,
             missing: str = 'last') -> Collection[EntityT]:
        """A collection like this one, read in the order of the given keys.

            rois.sort('index')                      # ascending
            rois.sort('-dff_max')                   # descending
            rois.sort('layer_index', '-dff_max')    # major, then minor
            rois.sort('id')                         # Roi_2 before Roi_10
            rois.sort('id', natural=False)          # Roi_10 before Roi_2
            rois.sort('snr', missing='first')

        Parent attributes work as keys, in either spelling:

            rois.sort('[Animal]id', '[Layer]index', 'index')

        The order is worked out in Python, from values read through the same
        pivot `dataframe_of` uses, rather than as a SQL ORDER BY. The backends
        do not agree about string order - SQLite compares bytes, MySQL 8
        defaults to a case-insensitive collation - and `ArchiveBackend` is
        SQLite, so an archive would otherwise sort differently from the
        entarchy it was exported from. Sorting here costs one read of the key
        columns and gives the same answer everywhere.

        uuid is always appended as a final key, so entities that tie on
        everything asked for still come back in a reproducible order - the same
        order every time for this entarchy and for an archive exported from it,
        though not for a different entarchy holding the same values, since
        uuids are not shared between them.

        Set operations and `where()` drop the sort, because the order of a union
        of two differently sorted collections has no meaning worth guessing at.
        Sort the result instead.

        Args:
            *keys: attribute names, each optionally prefixed with '-' for
                descending.
            natural: compare digit runs inside text as numbers, so `Roi_2`
                precedes `Roi_10`. On by default, because entity ids are
                numbered - `Roi_0` to `Roi_1299`, `plane0` to `plane4` - and
                nobody reading them means the order that puts `Roi_100` second.
                Applies only to text keys; a key that already has a numeric or
                temporal order is left in it. Pass False for byte order.
            missing: 'last' (default) or 'first', for entities that do not have
                the attribute. A stored NaN is not distinguished from a missing
                attribute and is placed with it.

        Returns:
            Collection: a new collection; this one is unchanged.
        """
        if len(keys) == 0:
            raise ValueError('sort() needs at least one attribute name to sort by.')

        if missing not in ('first', 'last'):
            raise ValueError(f'missing must be "first" or "last", not {missing!r}.')

        parsed = []
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(f'Sort keys are attribute names; got {key!r}.')
            descending = key.startswith('-')
            name = key[1:] if descending else key
            if len(name) == 0:
                raise ValueError('An empty sort key is not an attribute name.')
            parsed.append((name, descending))

        self._refuse_unsortable([name for name, _ in parsed])

        sorted_collection = self._derive(self._as_tree)
        sorted_collection._sort_keys = parsed
        sorted_collection._sort_natural = natural
        sorted_collection._sort_missing = missing

        return sorted_collection

    def sort_by_hierarchy(self, natural: bool = True,
                          missing: str = 'last') -> Collection[EntityT]:
        """A collection read down the tree: each ancestor's id, then its own.

            rois.sort_by_hierarchy()
            # the same as
            rois.sort('[Animal]id', '[Recording]id', '[Imaging]id',
                      '[Layer]id', 'id')

        Which is the order almost anything printed wants, and tedious enough to
        spell out that it tends not to be. The chain is read from the entarchy's
        own hierarchy, so it follows whatever levels the schema declares.

        Returns:
            Collection: a new collection; this one is unchanged.
        """
        path = _find_path(self.entarchy.hierarchy, self.entity_type.__name__)
        if path is None:
            raise LookupError(
                f'{self.entity_type.__name__} is not in this entarchy\'s '
                f'hierarchy, so it has no ancestors to sort by.')

        # The last entry of the path is this type itself, which addresses its
        #  own id rather than an ancestor's
        keys = [f'[{ancestor}]id' for ancestor in path[:-1]] + ['id']

        return self.sort(*keys, natural=natural, missing=missing)

    @property
    def order(self) -> str:
        """How this collection is read, in the words sort() would take.

        Says `uuid` when nothing was asked for, rather than nothing at all: the
        order is arbitrary but it is not absent, and a reader looking at the
        first ten rows deserves to know which ten they are.
        """
        if self._sort_keys is None:
            return 'uuid'

        return ', '.join(self.sort_keys)

    def _refuse_unsortable(self, names: list[str]) -> None:
        """Reject keys that cannot be put in an order, before any work is done.

        A blob is an encoded container - a magic number, a header, then raw
        buffers - so ordering by one would compare those bytes and look like it
        had worked. Better to say so while the caller is still looking at the
        line that asked for it.
        """
        stored_names = [_attribute_name_of(name) for name in names]
        types = self.entarchy.backend.get_attribute_data_types(stored_names)

        unknown = [given for given, stored in zip(names, stored_names)
                   if stored not in types]
        if len(unknown) > 0:
            raise AttributeError(
                f'Cannot sort by {unknown}: not stored on any entity.')

        blobs = [given for given, stored in zip(names, stored_names)
                 if 'blob' in types[stored]]
        if len(blobs) > 0:
            raise TypeError(
                f'Cannot sort by {blobs}: stored as blobs, and a blob is an '
                f'encoded container rather than a value with an order. Sort by '
                f'something derived from it instead.')

    def _ordered_rows(self) -> list[tuple[str, str]] | None:
        """The (uuid, id) rows of this collection in its sort order, or None.

        None means unsorted, which every caller reads as "ask the backend".
        """
        if self._sort_keys is None or self._sort_resolving:
            return None

        if self._sort_order is None:
            self._sort_resolving = True
            try:
                self._sort_order = self._resolve_sort()
            finally:
                self._sort_resolving = False

        return self._sort_order

    def _resolve_sort(self) -> list[tuple[str, str]]:
        """Work out the order once, by reading the key columns."""
        rows = self.entarchy.backend.get_collection_entities_by_slice(
            self, slice(None, None, None))
        if len(rows) == 0:
            return []

        names = [name for name, _ in self._sort_keys]
        frame = self.dataframe_of(names)

        # Columns are labelled by position rather than by name: the index is
        #  called 'uuid' and so is the tiebreaker, and pandas refuses to sort by
        #  a label that is both. Integers cannot collide with it.
        ordering = pd.DataFrame(index=frame.index)
        for position, (name, _) in enumerate(self._sort_keys):
            column = frame[name]

            # Only text is reordered by reading its digits as numbers; doing it
            #  to a column that already has a numeric or temporal order would
            #  stringify it and undo that order. Asked as "has its own order"
            #  rather than "is object", because the pivot hands back pandas
            #  string dtype for text rather than object.
            if self._sort_natural and not (
                    pd.api.types.is_numeric_dtype(column)
                    or pd.api.types.is_datetime64_any_dtype(column)):
                column = column.map(_natural_sort_key, na_action='ignore')

            ordering[position] = column

        # The tiebreaker. Without it, entities equal on every key asked for come
        #  back in whatever order the sort happened to leave them, and "the
        #  first ten by response" can change membership between runs.
        tiebreak = len(self._sort_keys)
        ordering[tiebreak] = ordering.index

        ordering = ordering.sort_values(
            by=list(range(tiebreak + 1)),
            ascending=[not descending for _, descending in self._sort_keys] + [True],
            na_position=self._sort_missing,
            kind='stable')

        id_of = dict(rows)
        if set(id_of) != set(ordering.index):
            raise RuntimeError(
                f'{self} produced {len(id_of)} entities but {len(ordering.index)} '
                f'rows of attributes; refusing to guess how they pair up.')

        return [(uuid, id_of[uuid]) for uuid in ordering.index]

    @property
    def is_sorted(self) -> bool:
        """Whether an order was asked for, as opposed to the backend's own."""
        return self._sort_keys is not None

    @property
    def sort_keys(self) -> list[str]:
        """The keys this collection was sorted by, as sort() would take them."""
        if self._sort_keys is None:
            return []

        return [f'-{name}' if descending else name
                for name, descending in self._sort_keys]

    def _rows(self, _slice: slice = None) -> list[tuple[str, str]]:
        """(uuid, id) for this collection in its order, optionally sliced.

        The one place that decides whether an access path follows a sort or the
        backend's uuid order, so that they cannot drift apart.
        """
        ordered = self._ordered_rows()

        if ordered is None:
            return self.entarchy.backend.get_collection_entities_by_slice(
                self, slice(None, None, None) if _slice is None else _slice)

        return ordered if _slice is None else ordered[_slice]

    def _as_endpoint(self):
        """What this collection is, as a link endpoint constraint would see it."""
        from .links import Endpoint

        if getattr(self, 'link_type', None) is not None:
            return Endpoint(link_type=self.link_type)

        return Endpoint(entity_type=self.entity_type.__name__)

    def links(self, link_type: str, *_string_expressions: str, within: bool = False,
              **_equalities) -> 'LinkCollection':
        """Links of a kind that touch this collection.

            rois.links('correlated')                # at least one end is a member
            rois.links('correlated', within=True)   # both ends are members
            rois.links('correlated', within=True, r=0.9)

        `within=True` is the collection against itself: the correlations *among*
        these ROIs rather than every correlation any of them takes part in.

        Membership becomes a subquery, so this composes with whatever filter the
        collection already carries and has no size limit - unlike spelling the
        members out as `@both.uuid IN (...)`, which binds one parameter per uuid
        per endpoint and gives up around sixteen thousand of them.

        Returns:
            LinkCollection: chainable, so `.where(...)`, `dataframe_of` and
            `map_async` all work on the result.
        """
        spec = self.entarchy.require_link_type(link_type)
        endpoint = self._as_endpoint()

        accepted = [spec.linker.accepts(endpoint), spec.linked.accepts(endpoint)]
        if not any(accepted):
            raise LinkTypeError(
                f'"{spec.name}" connects {spec.linker} to {spec.linked}, so a '
                f'collection of {endpoint} is at neither end of it.')

        if within:
            # Both ends must be members, which only means anything if both ends
            #  of the kind could be one. Otherwise the answer is always empty,
            #  and an empty answer looks like a fact rather than a mistake.
            if not all(accepted):
                raise LinkTypeError(
                    f'within=True asks for links whose two ends are both in this '
                    f'collection, but "{spec.name}" connects {spec.linker} to '
                    f'{spec.linked} and this is a collection of {endpoint}. Drop '
                    f'within, or use links_to() for links reaching the other end.')
            constraints = [(self, self)]
        else:
            constraints = [(self, None), (None, self)]

        as_tree = self.entarchy._build_filter_tree(_string_expressions, _equalities)

        return LinkCollection(self.entarchy, spec.name, as_tree, constraints)

    def links_to(self, other: 'Collection', link_type: str, *_string_expressions: str,
                 **_equalities) -> 'LinkCollection':
        """Links of a kind running between this collection and another.

            day2_rois.links_to(day1_rois, 'same_cell')
            rois.links_to(phases, 'mean_response', mean_dff=0.5)

        Which collection is the linker is worked out from the kind's endpoint
        types, so the arguments may be given in either order when those types
        differ. For a symmetric kind both storage orders are searched, since
        which end is which is an artifact of uuid ordering there.
        """
        from .links import orientation

        spec = self.entarchy.require_link_type(link_type)
        own, others = self._as_endpoint(), other._as_endpoint()

        if orientation(spec, own, others) == 'swapped':
            constraints = [(other, self)]
        else:
            constraints = [(self, other)]

        if spec.symmetric:
            linker_side, linked_side = constraints[0]
            constraints.append((linked_side, linker_side))

        as_tree = self.entarchy._build_filter_tree(_string_expressions, _equalities)

        return LinkCollection(self.entarchy, spec.name, as_tree, constraints)

    def matrix_from_links(self, other: Collection, link_type: str, value_name: str,
                          *_string_expressions: str, **_equalities) -> pd.DataFrame:
        """The links between this collection and another, as a matrix.

            r = day1_rois.matrix_from_links(day2_rois, 'same_cell', 'overlap')
            r = rois.matrix_from_links(rois, 'correlation', 'r')

        Rows are this collection in its own order, columns are `other` in
        theirs, and a pair with no link is NaN. The same as
        `ent.matrix_from_links(self, other, ...)`, which is where it is
        documented.
        """
        return self.entarchy.matrix_from_links(
            self, other, link_type, value_name, *_string_expressions, **_equalities)

    def __len__(self):
        if self._length is None:
            self._length = self.entarchy.backend.get_collection_count(self)
        return self._length

    def __repr__(self):
        if self.name is not None:
            return self.name
        if self.__class__.__name__ == 'Collection':
            return (f'{self.__class__.__name__}(entity_type=\'{self.entity_type.__name__}\', '
                    f'count={len(self)}, order=\'{self.order}\')')
        return f'{self.__class__.__name__}(count={len(self)}, order=\'{self.order}\')'

    def _repr_html_(self) -> str:
        """Rich representation for notebooks.

        Shows the entity type, the number of matches and the first few entities.
        Attribute values are not loaded: use preview() or dataframe_of() for those,
        so the cost is always something the user asked for.
        """
        try:
            count = len(self)
            title = self.name or self.entity_type.__name__

            rows = self._rows(slice(0, _HTML_PREVIEW_ROWS))

            if count == 0:
                body = ('<div style="color:#888;font-size:90%">no matching entities</div>')
            else:
                cells = ''.join(
                    f'<tr><td style="font-family:monospace;padding-right:12px">{html.escape(str(_id))}</td>'
                    f'<td style="font-family:monospace;color:#888">{html.escape(str(_uuid))}</td></tr>'
                    for _uuid, _id in rows)
                # Which rows these are depends entirely on the order, and an
                #  unsorted collection is in uuid order rather than in no order
                #  at all - so say which, especially when this is a truncated
                #  view of something much larger
                more = (f'<div style="color:#888;font-size:90%">'
                        f'showing {len(rows)} of {count}, ordered by '
                        f'<code>{html.escape(self.order)}</code></div>'
                        if count > len(rows) else
                        f'<div style="color:#888;font-size:90%">ordered by '
                        f'<code>{html.escape(self.order)}</code></div>')
                body = (f'<table style="border:none">'
                        f'<tr><th style="text-align:left;color:#888">id</th>'
                        f'<th style="text-align:left;color:#888">uuid</th></tr>'
                        f'{cells}</table>{more}')

            return (f'<div>'
                    f'<b>{html.escape(str(title))}</b> '
                    f'<span style="color:#888">&middot; {count} '
                    f'{html.escape(self.entity_type.__name__)} '
                    f'{"entity" if count == 1 else "entities"}</span>'
                    f'{body}'
                    f'<div style="color:#888;font-size:90%">'
                    f'<code>.preview()</code> for attribute values, '
                    f'<code>.dataframe_of([...])</code> for a full DataFrame</div>'
                    f'</div>')
        except Exception:
            # A repr must never break a notebook session
            return _fallback_html(self)

    def describe(self, links: bool = True,
                 distribution: bool = False) -> 'describe_mod.Description':
        """What this collection holds, asked of the set rather than of one entity.

            rois.describe()                 # renders whole
            rois.describe().attributes      # a DataFrame

        The attributes section gains the column a single entity cannot show:
        `entities`, how many members actually carry each name. Attributes are
        per entity rather than per type, so `ants/x` on 34 000 of 42 521 ROIs is
        a fact about the data worth meeting without going looking for it.

        Args:
            links: count the link kinds touching this collection. One query, but
                over every member, so it can be turned off for a large one.
            distribution: also report the lowest and highest value each scalar
                attribute holds and how many different ones there are. Off by
                default because it is a query per stored type on top of the one
                the section already costs, and because a range is a question
                about the data rather than about what is in there.

        Returns:
            Description: sections that are DataFrames, rendering together.
        """
        from . import describe as describe_mod

        backend = self.entarchy.backend
        notes = []
        count = len(self)

        try:
            metadata = backend.get_collection_attribute_metadata(self)
        except Exception as exc:
            metadata = []
            notes.append(f'attributes could not be read: {exc}')

        ranges = None
        if distribution and len(metadata) > 0:
            # Only the types this collection actually stores. Each is its own
            #  query and the entity subquery is the expensive part of every one
            present = sorted({data_type for _, data_type, _, _ in metadata
                              if data_type in describe_mod.SCALAR_TYPES})
            try:
                ranges = backend.get_collection_attribute_distribution(
                    self, data_types=present)
            except Exception as exc:
                notes.append(f'ranges could not be read: {exc}')

        if ranges:
            with_nan = sorted({name for (name, _), entry in ranges.items()
                               if entry['nan']})
            if len(with_nan) > 0:
                notes.append(
                    f'NaN is stored on {", ".join(with_nan)}. It counts as one '
                    f'more distinct value and is left out of the range, having '
                    f'no place in an ordering.')

        link_counts = {}
        if links and count > 0:
            try:
                link_counts = backend.count_collection_links_by_type(self)
            except Exception as exc:
                notes.append(f'link counts could not be read: {exc}')

        children = {}
        if count > 0:
            try:
                children = backend.count_collection_child_entities(self)
            except Exception as exc:
                notes.append(f'child counts could not be read: {exc}')

            # Link carriers are parented to their linker; they are links, and
            #  the links section is where they belong
            children.pop(describe_mod.LINK_ENTITY_TYPE, None)

        sections = {
            'attributes': describe_mod.collection_attribute_rows(
                metadata, count, distribution=ranges),
            'links': describe_mod.link_rows(link_counts),
            'children': describe_mod.count_rows(children, 'type', 'count'),
        }

        return describe_mod.Description(
            subject=self.name or f'{count} {self.entity_type.__name__}',
            headline={'entity type': self.entity_type.__name__,
                      'entities': count,
                      'order': self.order},
            sections=sections, notes=notes)

    def preview(self, n: int = 10, attribute_names: list[str] = None,
                blobs: bool = False) -> pd.DataFrame:
        """Return the first n entities as a DataFrame, for interactive inspection.

        Blobs are left out unless asked for, because a preview that reads them is
        not a preview: three rows of a Recording in the vxpy schema came to 653 MB
        of array data, and three rows of a Layer to 196 MB. Which names were left
        out is printed and recorded in `df.attrs['blobs_omitted']`, so their
        absence is visible rather than silent.

        Args:
            n (int): Number of entities to show.
            attribute_names (list of str): Attributes to include. Defaults to the
                scalar attributes of the collection. Named attributes are taken
                as asked for, blob or not.
            blobs (bool): Include blob attributes in the default selection.

        Returns:
            pandas.DataFrame
        """

        rows = self._rows(slice(0, n))
        if len(rows) == 0:
            return pd.DataFrame()

        subset = self.entarchy.get(self.entity_type,
                                   ' OR '.join(f'uuid == "{_uuid}"' for _uuid, _ in rows))

        omitted = []
        if attribute_names is None:
            attribute_names = [name for name in subset.columns if name != 'uuid']

            if not blobs:
                types = self.entarchy.backend.get_attribute_data_types(attribute_names)
                omitted = sorted(name for name in attribute_names
                                 if 'blob' in types.get(name, set()))
                attribute_names = [name for name in attribute_names
                                   if name not in set(omitted)]

        if len(omitted) > 0:
            shown = ', '.join(omitted[:5])
            more = f' and {len(omitted) - 5} more' if len(omitted) > 5 else ''
            print(f'preview: left out {len(omitted)} blob attribute(s) - {shown}{more}. '
                  f'Pass blobs=True or name them to read them.')

        # The subset is a fresh collection and so carries none of this one's
        #  order; put the rows back the way they were asked for
        frame = subset.dataframe_of(attribute_names).reindex(
            [_uuid for _uuid, _ in rows])
        frame.attrs['blobs_omitted'] = omitted

        return frame

    # Access methods

    @overload
    def __getitem__(self, item: int | np.integer) -> EntityT: ...

    @overload
    def __getitem__(self, item: slice) -> list[EntityT]: ...

    @overload
    def __getitem__(self, item: str) -> pd.Series: ...

    @overload
    def __getitem__(self, item: list[str]) -> pd.DataFrame: ...

    def __getitem__(self, item):
        # An index gives one entity and a slice a list of them; a name or a
        #  list of names gives values instead, as a Series or as a frame. The
        #  overloads above are how a reader of the signature - or an editor -
        #  gets told which of the four they asked for

        # Return single entity
        if isinstance(item, (int, np.integer)):
            if item < 0:
                item = len(self) + item

            ordered = self._ordered_rows()
            if ordered is None:
                _uuid, _id = self.entarchy.backend.get_collection_entity_by_index(self, item)
            else:
                if not 0 <= item < len(ordered):
                    raise IndexError(f'{self} has {len(ordered)} entities; '
                                     f'no index {item}.')
                _uuid, _id = ordered[item]

            return self.get_entity(_uuid=_uuid, _id=_id)

        # Return slice
        elif isinstance(item, slice):

            # Get data
            res = self._rows(item)

            result = [self.get_entity(_uuid=_uuid, _id=_id) for _uuid, _id in res]

            return result

        # Return multiple attributes for all entities in collection
        elif isinstance(item, (str, list)):
            if isinstance(item, str):
                item = [item]

            df = self.dataframe_of(attribute_names=item)

            # For single column, return pd.Series
            if len(df.columns) == 1:
                return df.iloc[:, 0]

            return df

        raise KeyError(f'Invalid key {item}')

    def __iter__(self) -> CollectionIterator[EntityT]:
        return CollectionIterator(self)

    def __setitem__(self, key, value):

        # if isinstance(key, slice) and key != slice(None, None, None):
        #     raise KeyError(f'Invalid key {key}')

        # Set series data to key attribute name
        if isinstance(value, pd.Series):
            value.name = key
            df = pd.DataFrame(value)

        # If value is a scalar, set whole collection to value for key
        elif isinstance(value, (str, int, float, bool)):
            df = pd.DataFrame(index=self.index, columns=[key], data=[value] * len(self.index))

        else:
            raise RuntimeError('Invalid key/value pair')

        self.update(df)

    # Set operations

    def __or__(self, other: Collection | str) -> Collection[EntityT]:
        """Create a new collection that is the union between this collection and another.

        Args:
            other (Collection | str): Another collection to add.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a + collection_b
        """

        # If other is a string, treat it as a query and add the resulting collection to this collection
        if isinstance(other, str):
            return self | self.entarchy.get(self.entity_type, other)

        if len(self.as_tree) == 0 or len(other.as_tree) == 0:
            raise RuntimeError('One of the AS trees is empty. '
                               'Cannot find union of universal set.')

        if not isinstance(other, Collection):
            raise TypeError('Can only add another Collection instance.')

        if self.entity_type != other.entity_type:
            raise ValueError('Can only add collections of the same entity type.')

        # Combine and return new collection
        new_tree = query.combine_trees('UNION', self.as_tree, other.as_tree)
        return self._derive(new_tree)

    def __and__(self, other: Collection | str) -> Collection[EntityT]:
        """Create a new collection that is the intersection between this collection and another.

        Args:
            other (Collection | str): Another collection to intersect.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a & collection_b
        """

        # If other is a string, treat it as a query and intersect with the resulting collection
        if isinstance(other, str):
            return self.where(other)

        if len(self.as_tree) == 0 or len(other.as_tree) == 0:
            raise RuntimeError('One of the AS trees is empty. '
                               'Cannot find intersection of universal set.')

        if not isinstance(other, Collection):
            raise TypeError('Can only intersect with another Collection instance.')

        if self.entity_type != other.entity_type:
            raise ValueError('Can only intersect collections of the same entity type.')

        # Combine and return new collection
        new_tree = query.combine_trees('INTERSECTION', self.as_tree, other.as_tree)
        return self._derive(new_tree)

    def __invert__(self) -> Collection[EntityT]:
        """Create a new collection that is the complement of this collection.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = ~collection_a
        """

        if len(self.as_tree) == 0:
            raise RuntimeError('AS tree is empty. '
                               'Cannot find complement of universal set.')

        # Invert and return new collection
        new_tree = query.combine_trees('COMPLEMENT', self.as_tree)
        return self._derive(new_tree)

    def __sub__(self, other: Collection | str) -> Collection[EntityT]:
        """Create a new collection that is the difference between this collection and another.

        Args:
            other (Collection | str): Another collection to subtract.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a - collection_b
        """

        if isinstance(other, str):
            return self - self.entarchy.get(self.entity_type, other)

        if len(self.as_tree) == 0 or len(other.as_tree) == 0:
            raise RuntimeError('One of the AS trees is empty. '
                               'Cannot find difference of universal set.')

        if not isinstance(other, Collection):
            raise TypeError('Can only subtract another Collection instance.')

        if self.entity_type != other.entity_type:
            raise ValueError('Can only subtract collections of the same entity type.')

        # Combine and return new collection
        new_tree = query.combine_trees('DIFFERENCE', self.as_tree, other.as_tree)
        return self._derive(new_tree)

    def __xor__(self, other: Collection | str) -> Collection[EntityT]:
        """Create a new collection that is the symmetric difference between this collection and another.

        Args:
            other (Collection | str): Another collection to xor.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a ^ collection_b
        """

        if isinstance(other, str):
            return self ^ self.entarchy.get(self.entity_type, other)

        if len(self.as_tree) == 0 or len(other.as_tree) == 0:
            raise RuntimeError('One of the AS trees is empty. '
                               'Cannot perform symmetric difference on universal set.')

        if not isinstance(other, Collection):
            raise TypeError('Can only xor with another Collection instance.')

        if self.entity_type != other.entity_type:
            raise ValueError('Can only xor collections of the same entity type.')

        # Combine and return new collection
        new_tree = query.combine_trees('SYMMETRIC_DIFFERENCE', self.as_tree, other.as_tree)
        return self._derive(new_tree)

    # Properties

    @property
    def as_tree(self) -> dict[str, Any]:
        """Return the abstract syntax tree dictionary
        """
        return self._as_tree.copy()

    @property
    def columns(self):
        return self.entarchy.backend.get_collection_attribute_names(self)

    @property
    def entarchy(self) -> Entarchy:
        return self._entarchy

    @property
    def entity_type(self) -> Type[EntityT]:
        return self._entity_type

    @property
    def index(self):

        # If cache is empty, initialize it. Otherwise, the uuid index won't be there.
        if len(self._cache.index) == 0:
            # TODO: there is a bug here. If uuid is called first, it does not return any attribute,
            #  because the returned DataFrame just contains the uuid index and no columns
            #  -> look into this
            self._load_attributes(['id'])

        return self._cache.index

    @property
    def init_time(self) -> datetime.datetime:
        return self._init_time

    @property
    def name(self):
        return self._name

    def _load_attributes(self, attribute_names: list[str]):

        # Load attributes from backend
        df = self.entarchy.backend.get_collection_attributes(self, attribute_names)

        # Update cache
        self._cache[df.columns] = df

    def dataframe_of(self, attribute_names: list[str] = None, reload_cached: bool = False) -> pd.DataFrame:

        # Default to all attributes of the collection
        if attribute_names is None:
            attribute_names = self.columns

        original_attribute_order = list(attribute_names)

        # Check for parent attributes in attribute_names and separate them from regular attributes,
        #  because they need to be fetched separately.
        # 'uuid' is the cache index, not a column - handle it as a derived column below.
        parent_attribute_names = [n for n in attribute_names if n.startswith('../') or n.startswith('[')]
        uuid_requested = 'uuid' in attribute_names
        attribute_names = list(set(attribute_names) - set(parent_attribute_names) - {'uuid'})

        # Make sure the cache index (entity uuids) is initialized even if no
        #  regular attribute triggers a load (e.g. only 'uuid' was requested)
        if len(self._cache.index) == 0 and len(attribute_names) == 0:
            self._load_attributes(['id'])

        # If not all attributes are in cache, fetch the missing ones
        loaded_attributes = set(attribute_names) & set(self._cache.columns.tolist())
        if reload_cached or (len(loaded_attributes) < len(attribute_names)):

            # Check which attributes to load
            if not reload_cached:
                _attributes_cached = list(set(attribute_names) & set(self._cache.columns.tolist()))
                _attributes_to_fetch = list(set(attribute_names) - set(self._cache.columns.tolist()))
            else:
                _attributes_cached = []
                _attributes_to_fetch = attribute_names

            if self._entarchy.debug:
                print('Cached attributes:', _attributes_cached)
                print('Attributes to fetch: ', _attributes_to_fetch)

            # Load missing attributes into cache
            self._load_attributes(_attributes_to_fetch)

        # If there were parent attributes selected, fetch them individually
        # - Parent attributes can either be accessed relative to the current entity type level using "../"
        #    or by specifying the parent entity type in square brackets ("[ParentEntityTypeName]attr_name")
        # - Parent attributes are not cached in the collection cache,
        #    because they are not needed for most operations and would cause too much overhead
        if len(parent_attribute_names) > 0:

            parent_attribute_names_to_fetch = parent_attribute_names.copy()

            # print('Get parent attributes:', parent_attribute_names_to_fetch)
            parent_df = pd.DataFrame(index=self._cache.index)

            # Which parent each entity has, keyed by the entity rather than
            #  taken in the order it arrives. This query and the pivot that
            #  filled the cache are ordered independently of one another - the
            #  pivot groups without an ORDER BY, and MySQL 8 dropped the
            #  implicit ordering of GROUP BY that SQLite happens to give - so
            #  pairing them off by position was trusting them to agree. Where
            #  they did not, every parent value landed on the wrong entity.
            parent_of = dict(self.entarchy.backend.get_collection_parent_uuids(self))

            for parent_attr in parent_attribute_names_to_fetch:

                if parent_attr.startswith('../'):
                    parent_level = parent_attr.count('../')
                    parent_attr_name = parent_attr.replace('../', '')
                else:
                    if not (('[' in parent_attr) and (']' in parent_attr)):
                        raise ValueError('Malformed attribute name for parent entity traversal. '
                                         'Explicit parent attribute addressing must be '
                                         'of format "[ParentEntityTypeName]attr_name".')
                    parent_entity_type_name, parent_attr_name = parent_attr.replace('[', '').split(']')
                    parent_level = get_ancestor_distance_from_nested(self.entarchy.hierarchy,
                                                                     self.entity_type.__name__,
                                                                     parent_entity_type_name)

                # Create a list of parent values, in the order of the rows they
                #  are about
                parent_values = []
                for entity_uuid in self._cache.index:
                    parent_uuid = parent_of.get(entity_uuid)

                    _parent_value = None
                    parent_entity = None
                    if parent_uuid is not None:
                        parent_entity = self.entarchy.get_entity_by_uuid(parent_uuid)

                    # Traverse parent hierarchy
                    if parent_entity is not None:
                        for _ in range(parent_level - 1):

                            if parent_entity.parent is None:
                                _parent_value = None
                                break

                            parent_entity = parent_entity.parent

                        # Get parent attribute value
                        _parent_value = parent_entity[parent_attr_name]

                    # Add parent value to list
                    parent_values.append(_parent_value)

                # Add whole column to parent_df
                parent_df[parent_attr] = parent_values

            df = pd.concat([self._cache[attribute_names], parent_df], axis=1, copy=True)

            if uuid_requested:
                df['uuid'] = df.index

            return self._in_sort_order(df[original_attribute_order])

        df = self._cache[attribute_names].copy()

        if uuid_requested:
            df['uuid'] = df.index

        return self._in_sort_order(df[original_attribute_order])

    def _in_sort_order(self, df: pd.DataFrame) -> pd.DataFrame:
        """Put a frame in the collection's order, if one was asked for.

        The cache is indexed by uuid, so this is a reindex rather than a second
        read - which is what makes sorting in Python affordable here.
        """
        ordered = self._ordered_rows()
        if ordered is None:
            return df

        return df.reindex([_uuid for _uuid, _ in ordered])

    def get_entity(self, _uuid: str, _id: str) -> EntityT:

        _init_cache = None
        if _uuid in self._cache.index:
            _init_cache = self._cache.loc[_uuid].to_dict()

        return self.entity_type(_entarchy=self.entarchy, _uuid=_uuid, _id=_id, _init_cache=_init_cache)

    def keys(self) -> list[str]:
        return self.columns

    def _ipython_key_completions_(self) -> list[str]:
        """The stored names, for tab completion inside the brackets.

        IPython asks an object for this when the cursor is inside a subscript.
        Nothing here defined it, so there was nothing entarchy-specific to
        offer and the completer fell back to the global namespace: tabbing
        inside an entity's brackets suggested 335 builtins and magics and not
        one of the names actually stored.

        They cannot be reached any other way. `s2p/npix` is a row in the
        attributes table rather than a Python attribute, so `dir()` cannot see
        it and no static tool can know it - but one indexed query can.
        """
        try:
            return self.keys()
        except Exception:
            # This runs on a keystroke. An entity whose backend has gone away
            #  should still be typeable, so a completer must never raise
            return []

    def map(self, fun: Callable, **kwargs) -> list:
        """Sequentially apply a function to each Entity of the collection (kwargs are passed onto the function)

        Returns:
            list: the return values of fun for each entity, in collection order.
        """

        entity_count = len(self)

        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {entity_count} entities')

        results = []
        with alive_progress.alive_bar(entity_count, length=30, force_tty=True,
                                      **console.bar_style()) as bar:
            for entity in self:
                results.append(fun(entity, **kwargs))
                bar()

        return results

    def map_async(self,
                  _fun: Callable,
                  _worker_num: int = None,
                  _chunk_size: int = None,
                  _use_gpu: bool = False,
                  _gpu_max_device_num: int = None,
                  _calibrate: bool = True,
                  _locality: bool = None,
                  **kwargs) -> None:
        """Apply a function to each Entity of the collection in parallel worker processes.

        Results are not collected; the function is expected to write what it computes
        back onto the entity. Entities are released from the registry after being
        processed, so memory does not grow with the size of the collection.

        Workers are kept alive between calls, so a sequence of map_async calls pays
        the pool startup cost once. Call shutdown_worker_pool() to release them early.

        If the function raises for individual entities, the remaining entities are
        still processed; a summary is printed at the end and a RuntimeError carrying
        the first traceback is raised.

        Args:
            _fun (Callable): Function to apply. It receives one Entity as its first
                argument, followed by **kwargs. It must be importable by name, because
                workers are started with the 'spawn' method.
            _worker_num (int): Number of worker processes (default: CPU count - 2,
                at least 1, never more than the number of entities).
            _chunk_size (int): Number of tasks handed to a worker at a time.
            _use_gpu (bool): Pass a '_use_gpu_device' keyword to the function, pinning
                one CUDA device per worker.
            _gpu_max_device_num (int): Number of CUDA devices to distribute across
                (default: torch.cuda.device_count()).
            _calibrate (bool): Measure the first few entities in this process and skip
                the worker pool when starting it would cost more than it saves. Set to
                False to always use the pool.
            _locality (bool): Group entities by parent, so a worker processes entities
                that share a parent together and can reuse the parent's cached
                attributes. Set to False to keep the collection's own order.
                Defaults to True, except on a sorted collection, where grouping
                would undo the order that was asked for.
            **kwargs: Passed on to the function.

        Returns:
            None
        """

        import multiprocessing as mp

        # Address entities by UUID rather than shipping pickled Entity objects
        entity_rows = self._rows()
        entity_count = len(entity_rows)

        print(f'Run function {_fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {entity_count} {self.entity_type.__name__} entities')

        if entity_count == 0:
            print('No entities to operate on in collection')
            return

        # Worker count must stay positive; cpu_count() - 2 is <= 0 on small machines
        if _worker_num is None:
            _worker_num = mp.cpu_count() - 2
        _worker_num = max(1, min(int(_worker_num), entity_count))

        # Anything defined in a notebook cell or the REPL lives in a __main__ that
        #  spawned workers cannot import. Pickled by reference it makes every worker
        #  die while unpickling, and the pool replaces them forever, so the session
        #  hangs with no error. Ship such objects by value, or stay in this process.
        interactive = [obj for obj in (self.entarchy, self.entity_type, _fun)
                       if _defined_interactively(obj)]
        by_value = False
        if interactive:
            if _cloudpickle_available():
                by_value = True
            else:
                names = ', '.join(sorted({getattr(o, '__name__', type(o).__name__)
                                          for o in interactive}))
                print(f'WARNING: {names} defined interactively and cloudpickle is not '
                      f'installed, so worker processes could not reconstruct it.')
                print(f'         Running in this process instead. To use {_worker_num} '
                      f'workers, either "pip install cloudpickle" or move the definitions '
                      f'into an importable module.')
                _worker_num = 1

        # Group entities by parent, so a worker sees runs of entities that share
        #  the (often large) attributes of their parent instead of jumping between
        #  parents in UUID order. Entities are stored with random UUID4 keys, so
        #  the unsorted order has no locality at all.
        #
        #  A sorted collection is the exception: regrouping would silently undo
        #  the order that was asked for, so locality is off by default there and
        #  has to be asked for. Passing it explicitly still wins either way.
        if _locality is None:
            _locality = not self.is_sorted

        parent_uuids = {}
        if _locality:
            try:
                parent_uuids = dict(self.entarchy.backend.get_collection_parent_uuids(self))
            except Exception as e:
                print(f'WARNING: could not determine parent entities for grouping ({e}); '
                      f'processing in default order')
                parent_uuids = {}

            if parent_uuids:
                entity_rows = sorted(entity_rows,
                                     key=lambda row: (parent_uuids.get(row[0]) or '', row[0]))

        # Resolve the GPU device count whenever GPU use is requested, not only when
        #  the count was left unset - otherwise passing it explicitly disabled the GPU
        gpu_device_count = None
        if _use_gpu:
            if _gpu_max_device_num is None:
                import torch
                _gpu_max_device_num = torch.cuda.device_count()
            gpu_device_count = int(_gpu_max_device_num)

        # Chunk size decides how tasks are handed out. When entities are grouped by
        #  parent, chunks that match the group size let a worker finish a whole group
        #  before moving on; chunks that straddle group boundaries make workers
        #  release a parent and load it again.
        if _chunk_size is None:
            if parent_uuids:
                group_sizes = {}
                for _parent in parent_uuids.values():
                    group_sizes[_parent] = group_sizes.get(_parent, 0) + 1
                sizes = sorted(group_sizes.values())
                typical_group = sizes[len(sizes) // 2]

                # Cap so that the work still spreads over all workers
                _chunk_size = max(1, min(typical_group, entity_count // _worker_num))
            else:
                _chunk_size = max(1, entity_count // (_worker_num * POOL_CHUNKS_PER_WORKER))

        start_time = time.time()
        print(f'Start processing at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

        failures: list[tuple[str, str]] = []

        with alive_progress.alive_bar(entity_count, length=20, force_tty=True,
                                      **console.bar_style(spinner_length=20)) as progress_bar:

            remaining = entity_rows

            # Estimate the cost of the work before paying for a process pool.
            #  Skipped for GPU runs, which are heavy by definition and should not be
            #  sampled on the CPU of the parent process.
            if _calibrate and gpu_device_count is None and _worker_num > 1:
                # A pool that is already running costs almost nothing to use again
                pool_is_warm = _POOL_CACHE.get('key') == _worker_pool_key(self.entarchy, _worker_num)
                startup_cost = (POOL_WARM_STARTUP_SECONDS if pool_is_warm
                                else POOL_STARTUP_ESTIMATE_SECONDS)

                sample_size = min(POOL_CALIBRATION_SAMPLE, entity_count)
                sample_start = time.perf_counter()
                sampled = 0
                estimated_sequential = estimated_parallel = 0.0

                # Sampled entities are processed, not wasted, but they run serially.
                #  Stop as soon as the pool is a clear win, so long jobs pay for one
                #  sample rather than the full set.
                while sampled < sample_size:
                    _uuid, _id = remaining[sampled]
                    error = self._apply_to_entity(_fun, _uuid, _id, kwargs)
                    if error is not None:
                        failures.append((_uuid, error))
                    progress_bar()
                    sampled += 1

                    seconds_per_entity = (time.perf_counter() - sample_start) / sampled
                    estimated_sequential = seconds_per_entity * (entity_count - sampled)
                    estimated_parallel = startup_cost + estimated_sequential / _worker_num

                    if estimated_parallel < 0.8 * estimated_sequential:
                        break

                remaining = remaining[sampled:]

                if remaining and estimated_parallel >= estimated_sequential:
                    print(f'Estimated {estimated_sequential:.1f}s of remaining work '
                          f'({1000 * seconds_per_entity:.1f}ms per entity): running in this '
                          f'process, because using {_worker_num} workers would cost more '
                          f'than it saves')
                    for _uuid, _id in remaining:
                        error = self._apply_to_entity(_fun, _uuid, _id, kwargs)
                        if error is not None:
                            failures.append((_uuid, error))
                        progress_bar()
                    remaining = []

            if remaining and _worker_num == 1:
                for _uuid, _id in remaining:
                    error = self._apply_to_entity(_fun, _uuid, _id, kwargs)
                    if error is not None:
                        failures.append((_uuid, error))
                    progress_bar()
                remaining = []

            if remaining:
                pool, was_running = _get_worker_pool(self.entarchy, _worker_num, by_value)
                print(f'{"Reuse" if was_running else "Start"} worker pool '
                      f'({_worker_num} workers, spawn, chunk size {_chunk_size}'
                      f'{", by value" if by_value else ""})')

                # The function and its arguments are identical for every task, so
                #  they are serialized once and shared by all of them
                job = _for_workers((_fun, tuple(kwargs.items())), by_value)
                entity_type_name = self.entity_type.__name__

                tasks = [(_uuid, _id, parent_uuids.get(_uuid), entity_type_name,
                          job, gpu_device_count)
                         for _uuid, _id in remaining]

                try:
                    for _uuid, error in pool.imap_unordered(_run_map_worker, tasks,
                                                            chunksize=_chunk_size):
                        if error is not None:
                            failures.append((_uuid, error))
                        progress_bar()
                except BaseException:
                    # A broken pool cannot be reused; make sure the next call rebuilds it
                    shutdown_worker_pool()
                    raise

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(f'\nFinish processing at {formatted_time}')

        if failures:
            print(f'\n{len(failures)} of {entity_count} entities failed. '
                  f'Successful entities have been committed; re-run to retry the rest '
                  f'(a filter such as \'NOT(EXIST(<attribute>))\' selects them).')
            for _uuid, _ in failures[:5]:
                print(f'  failed: {_uuid}')
            if len(failures) > 5:
                print(f'  ... and {len(failures) - 5} more')

            raise RuntimeError(f'{_fun.__name__} failed on {len(failures)} of {entity_count} '
                               f'entities. First traceback:\n\n{failures[0][1]}')

    def rename(self, name: str) -> None:
        self._name = name

    def _apply_to_entity(self, fun: Callable, _uuid: str, _id: str,
                         kwargs: dict) -> Union[str, None]:
        """Apply the mapped function to one entity in the current process.

        Returns the formatted traceback if the function raised, otherwise None.
        """

        # Entities the caller already holds stay registered, so their identity is
        #  preserved; entities materialized here are released again, so a large
        #  in-process run does not accumulate the whole collection in memory
        was_registered = _uuid in self.entarchy

        entity = self.entity_type(self.entarchy, _uuid=_uuid, _id=_id)
        try:
            with entity:
                fun(entity, **kwargs)
            return None
        except Exception:
            return traceback.format_exc()
        finally:
            if not was_registered:
                self.entarchy.forget_entity(entity)

    def to_asdf(self, destination: str, **kwargs) -> dict:
        """Write this collection, and its ancestors, to an ASDF archive.

        The archive is itself an entarchy, so `Entarchy(destination)` opens it
        and every query and DataFrame call works there unchanged. Ancestors come
        along so that parent lookups still resolve. See entarchy.tools.archive.
        """
        from ..tools import archive

        return archive.export(self.entarchy, destination, collection=self, **kwargs)

    def to_dict(self) -> Generator[dict[str, Any], None, None]:
        """
        Return a generator of dictionaries of all dynamic attributes for each entity in the collection.
        """
        for entity in self:
            yield entity.to_dict()

    def update(self, df: pd.DataFrame):

        # TODO: idea - allow update using individual records (i.e., list of dicts) to avoid pandas dependency here

        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas DataFrame.')

        # Validate names
        if not all(isinstance(n, str) for n in df.columns):
            raise TypeError('Attribute names must be strings.')

        if any(n.startswith('../') or n.startswith('[') for n in df.columns):
            raise RuntimeError('Attempted illegal operation. '
                               'Cannot update parent attributes through collection of child items '
                               '(attributes starting with "../" or "[EntityTypeName]").')

        # Send to backend first - if it rejects the update, the cache stays
        #  consistent with the database
        self.entarchy.backend.set_collection_attributes(self, df)

        # Update cache
        self._cache.update(df)

    def where(self, *_string_expressions: str, **_equalities) -> Collection[EntityT]:
        _collection = self.entarchy.get(self.entity_type, *_string_expressions, **_equalities)
        if self.as_tree == {}:
            new_tree = _collection.as_tree
        else:
            new_tree = query.combine_trees('INTERSECTION', self.as_tree, _collection.as_tree)

        return self._derive(new_tree)


class LinkCollection(Collection[LinkEntity]):
    """A queryable set of links of one kind, optionally with its ends pinned.

    Everything Collection does works here - slicing, dataframe_of, update,
    map_async, to_asdf - because a link is an entity. What it adds is the kind
    filter and the endpoint syntax in filter expressions:

        ent.links('mean_response',
                  '@Phase.index == 3 AND @Roi.has_receptive_field == True '
                  'AND mean_dff > 0.3')

    A bare name is an attribute of the link itself; `@` addresses an endpoint.
    """

    def __init__(self, _entarchy, _link_type: str, _as_tree,
                 _endpoint_constraints: list = None):
        Collection.__init__(self, _entarchy, LinkEntity, _as_tree)
        self._link_type = _link_type

        # Pairs of (linker collection, linked collection), either side possibly
        #  None for "anything", OR-ed together. Membership is applied as a
        #  subquery rather than a list of uuids, so it composes with the member
        #  collection's own filter and is not bounded by how many parameters a
        #  statement may bind.
        self._endpoint_constraints = list(_endpoint_constraints or [])

    @property
    def link_type(self) -> str:
        return self._link_type

    @property
    def endpoint_constraints(self) -> list:
        return self._endpoint_constraints

    @property
    def link_type_spec(self):
        spec = self.entarchy.get_link_type(self._link_type)
        if spec is None:
            from .links import LinkTypeError

            raise LinkTypeError(f'Link type "{self._link_type}" is not defined.')

        return spec

    def _derive(self, _as_tree) -> LinkCollection:
        return LinkCollection(self.entarchy, self._link_type, _as_tree,
                              self._endpoint_constraints)

    def __repr__(self):
        if self.name is not None:
            return self.name
        pinned = ' (endpoints restricted)' if self._endpoint_constraints else ''
        return (f'LinkCollection(\'{self._link_type}\'{pinned}, '
                f'count={len(self)}, order=\'{self.order}\')')


# Per-worker state for Collection.map_async.
#  Built once per worker process by _init_map_worker, so that the Entarchy object
#  and its database connection are reused across all tasks handled by that worker.
_WORKER_CONTEXT: dict[str, Any] = {}

# At most one reusable worker pool is kept alive, so repeated map_async calls do
#  not each pay the pool startup cost.
_POOL_CACHE: dict[str, Any] = {}

# Limits for the notebook representations, which must stay cheap: they run
#  automatically whenever an object is the last expression in a cell.
_HTML_MAX_ATTRIBUTES = 40
_HTML_PREVIEW_ROWS = 5
_HTML_MAX_LINK_TYPES = 12

# Wide enough for any count of anything an entarchy is likely to hold, and the
#  padding only has to be consistent within one sort to be correct.
_NATURAL_DIGIT_WIDTH = 20
_NATURAL_DIGIT_RUN = re.compile(r'\d+')


def _natural_sort_key(value: Any) -> str:
    """A string that orders the way a reader expects: Roi_2 before Roi_10.

    Digit runs are zero padded to a fixed width, so what comes back is still an
    ordinary string. A tuple of alternating text and numbers would order just as
    well until two ids disagree about their shape - `Roi_2` against `Roi_2b` -
    and then comparing int against str raises.
    """
    return _NATURAL_DIGIT_RUN.sub(
        lambda match: match.group(0).zfill(_NATURAL_DIGIT_WIDTH), str(value))


def _attribute_name_of(key: str) -> str:
    """The stored name behind a key, which may address a parent.

    `../depth` and `[Layer]depth` are both stored as `depth`, on the parent.
    """
    if key.startswith('../'):
        return key.replace('../', '')

    if key.startswith('[') and ']' in key:
        return key.split(']', 1)[1]

    return key


def _links_row_html(counts: dict[str, int]) -> str:
    """The link kinds row of the entity repr, or nothing when there are none."""
    if not counts:
        return ''

    shown = list(counts)[:_HTML_MAX_LINK_TYPES]
    listing = ', '.join(f'{html.escape(str(name))} '
                        f'<span style="color:#888">({counts[name]})</span>'
                        for name in shown)
    if len(counts) > len(shown):
        listing += (f' <span style="color:#888">... and '
                    f'{len(counts) - len(shown)} more</span>')

    label = f'{len(counts)} link {"kind" if len(counts) == 1 else "kinds"}'

    return (f'<tr><td style="text-align:right;color:#888;padding-right:8px;'
            f'vertical-align:top">{label}</td><td>{listing}</td></tr>')


def _fallback_html(obj: Any) -> str:
    """Last resort for _repr_html_, when even repr() cannot be produced."""
    try:
        text = repr(obj)
    except Exception:
        text = f'<{type(obj).__name__} (repr failed)>'
    return f'<pre>{html.escape(text)}</pre>'

# Rough cost of starting a process pool (spawn plus re-importing numpy/pandas/
#  sqlalchemy in each worker), used to decide whether parallel execution can pay
#  off at all. Measured at ~1.9 s; adjust if your environment differs markedly.
POOL_STARTUP_ESTIMATE_SECONDS = 2.0

# Cost of dispatching to a pool that is already running
POOL_WARM_STARTUP_SECONDS = 0.05

# Number of entities processed in-process to estimate the per-entity cost
POOL_CALIBRATION_SAMPLE = 3

# Chunks handed to each worker. More chunks balance load better, fewer keep
#  entities that share a parent together in one worker.
POOL_CHUNKS_PER_WORKER = 8


def _defined_interactively(obj: Any) -> bool:
    """Was this object defined in an interactive session (a notebook cell or REPL)?

    Such objects live in a __main__ that worker processes cannot import, so
    pickling them by reference produces a worker that dies while unpickling.
    """
    main_module = sys.modules.get('__main__')
    if main_module is not None and hasattr(main_module, '__file__'):
        # __main__ is a real script, which spawned workers can re-import
        return False

    # Instances resolve __module__ through their class
    return getattr(obj, '__module__', None) == '__main__'


def _cloudpickle_available() -> bool:
    try:
        import cloudpickle  # noqa: F401
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=8)
def _loads_by_value(data: bytes) -> Any:
    import cloudpickle

    return cloudpickle.loads(data)


class _ByValue:
    """Ships an object to workers by value instead of by reference.

    multiprocessing pickles by reference, which fails for anything defined in a
    notebook cell. Wrapping it here serializes the definition itself, so workers
    can reconstruct it without importing __main__.
    """
    __slots__ = ('_data',)

    def __init__(self, payload: Any):
        import cloudpickle

        # Serialized once in the parent; every task then ships the same bytes
        self._data = cloudpickle.dumps(payload)

    def __reduce__(self):
        return _loads_by_value, (self._data,)


def _for_workers(payload: Any, by_value: bool) -> Any:
    """Wrap a payload if it has to travel by value."""
    return _ByValue(payload) if by_value else payload


def _init_map_worker(_entarchy: Entarchy, _worker_counter) -> None:
    """Pool initializer: set up one Entarchy per worker process.

    Only state that stays valid across map_async calls is bound here, so the pool
    can be reused. The function, its arguments and the entity type travel with
    each task.
    """
    worker_index = 0
    if _worker_counter is not None:
        with _worker_counter.get_lock():
            worker_index = _worker_counter.value
            _worker_counter.value += 1

    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update({
        'entarchy': _entarchy,
        'worker_index': worker_index,
        'last_parent_uuid': None,
    })

    atexit.register(_close_map_worker)


def _close_map_worker() -> None:
    """Release the worker's database connection when the pool shuts down."""
    _entarchy = _WORKER_CONTEXT.get('entarchy')
    if _entarchy is None:
        return

    # Access the attribute, not the property, so a worker that never touched the
    #  database does not open a connection just to close it
    backend = _entarchy._backend
    if backend is not None:
        try:
            backend.close()
        except Exception:
            pass


def _run_map_worker(task: tuple) -> tuple:
    """Apply the mapped function to a single entity, addressed by UUID.

    Returns (uuid, traceback or None); failures are reported back rather than
    raised, so one bad entity does not abandon the rest of the collection.
    """
    _uuid, _id, parent_uuid, entity_type_name, job, gpu_device_count = task

    # A _ByValue payload has already turned back into the plain tuple here
    fun, kwargs_items = job

    _entarchy = _WORKER_CONTEXT['entarchy']

    # Resolve the type through the entarchy rather than shipping the class, so the
    #  worker uses the same class object as the rest of its entity map
    entity_type = _entarchy.get_entity_type(entity_type_name)

    kwargs = dict(kwargs_items)

    # Derive the device from the worker index, so a worker stays on one device
    if gpu_device_count is not None:
        kwargs['_use_gpu_device'] = (f'cuda:{_WORKER_CONTEXT["worker_index"] % gpu_device_count}'
                                     if gpu_device_count > 0 else 'cpu')

    entity = entity_type(_entarchy, _uuid=_uuid, _id=_id)
    error = None
    try:
        with entity:
            fun(entity, **kwargs)
    except Exception:
        error = traceback.format_exc()
    finally:
        # Without this the worker's registry would grow with every entity it handles
        _entarchy.forget_entity(entity)

        # Tasks are ordered by parent, so once the parent changes the previous one
        #  will not be needed again. Releasing it frees its cached attributes,
        #  which for imaging data can be hundreds of megabytes per parent.
        if parent_uuid is not None and parent_uuid != _WORKER_CONTEXT['last_parent_uuid']:
            if _WORKER_CONTEXT['last_parent_uuid'] is not None:
                _entarchy.forget_entity(_WORKER_CONTEXT['last_parent_uuid'])
            _WORKER_CONTEXT['last_parent_uuid'] = parent_uuid

    return _uuid, error


def _worker_pool_key(_entarchy: Entarchy, worker_num: int) -> tuple:
    """Identity of a pool. State that workers copy but cannot see change is part
    of the key, so a stale pool is rebuilt rather than silently reused."""
    analysis = _entarchy.current_analysis
    return (_entarchy.path,
            worker_num,
            analysis.uuid if analysis is not None else None,
            _entarchy.is_in_digest_mode)


def _get_worker_pool(_entarchy: Entarchy, worker_num: int, by_value: bool = False) -> tuple:
    """Return (pool, was_already_running) for this entarchy and worker count."""
    import multiprocessing as mp

    key = _worker_pool_key(_entarchy, worker_num)
    if _POOL_CACHE.get('key') == key and _POOL_CACHE.get('pool') is not None:
        return _POOL_CACHE['pool'], True

    # Only one pool is kept alive at a time
    shutdown_worker_pool()

    ctx = mp.get_context('spawn')
    counter = ctx.Value('i', 0)
    pool = ctx.Pool(processes=worker_num,
                    initializer=_init_map_worker,
                    initargs=(_for_workers(_entarchy, by_value), counter))

    _POOL_CACHE.update({'key': key, 'pool': pool})
    return pool, False


def shutdown_worker_pool() -> None:
    """Shut down the reusable map_async worker pool, if one is running.

    Called automatically at interpreter exit. Call it explicitly to release
    worker processes and their database connections early.
    """
    pool = _POOL_CACHE.pop('pool', None)
    _POOL_CACHE.pop('key', None)

    if pool is None:
        return

    try:
        # close() rather than terminate(), so workers run their exit handlers
        #  and close their database connections
        pool.close()
        pool.join()
    except Exception:
        pass


atexit.register(shutdown_worker_pool)


class CollectionIterator(Generic[EntityT]):

    def __init__(self, _collection: Collection[EntityT]):
        self._collection = _collection
        self._current_index = 0
        self._results = self._collection._rows()

    def __iter__(self) -> CollectionIterator[EntityT]:
        # The iterator protocol wants both halves on the iterator itself. Only
        #  __next__ was here, which is enough for a for-loop and not enough for
        #  iter(iter(collection)) or for a type checker to call this iterable
        return self

    def __next__(self) -> EntityT:
        # No more results: reset iteration counter and offset and stop iteration
        if self._current_index >= len(self._results):
            raise StopIteration

        # Return single result
        current_row = self._results[self._current_index]

        # Increment count
        self._current_index += 1

        return self._collection.get_entity(_uuid=current_row[0], _id=current_row[1])


def _find_path(hierarchy: dict[str, Any], target: str) -> list[str] | None:
    """
    Return path from current hierarchy root to target as a list of names, or None if not found.
    """
    for name, subtree in hierarchy.items():
        if name == target:
            return [name]
        if isinstance(subtree, dict):
            subpath = _find_path(subtree, target)
            if subpath is not None:
                return [name] + subpath
    return None


def get_ancestor_distance_from_nested(hierarchy: dict[str, Any],
                                      entity_type_name: str,
                                      parent_entity_type_name: str) -> int | None:
    """
    Return number of parent steps from entity_type_name up to parent_entity_type_name.
    Return 0 if same, or None if parent_entity_type_name is not an ancestor or either is missing.
    """
    if entity_type_name == parent_entity_type_name:
        return 0

    descendant_path = _find_path(hierarchy, entity_type_name)
    if descendant_path is None:
        return None

    ancestor_path = _find_path(hierarchy, parent_entity_type_name)
    if ancestor_path is None:
        return None

    # ancestor must be a prefix of descendant path
    if len(ancestor_path) > len(descendant_path):
        return None
    if descendant_path[: len(ancestor_path)] == ancestor_path:
        return len(descendant_path) - len(ancestor_path)
    return None


class DeferredEntityCollection(Generic[EntityT]):
    _expression: str

    def __init__(self, _entity_type: type[EntityT], _expression: str = ''):
        self._entity_type = _entity_type
        self._expression = _expression

    def __repr__(self):
        return (f'DeferredEntityCollection('
                f'entity_type=\'{self._entity_type.__name__}\', '
                f'expression=\'{self._expression}\''
                f')')

    def __and__(self, other: str | DeferredEntityCollection):
        return DeferredEntityCollection(self._entity_type, f'({self._expression}) AND ({self._get_relexpr(other)})')

    def __or__(self, other: str):
        return DeferredEntityCollection(self._entity_type, f'({self._expression}) OR ({self._get_relexpr(other)})')

    def __invert__(self):
        return DeferredEntityCollection(self._entity_type, f'NOT({self._expression})')

    def __sub__(self, other):
        return DeferredEntityCollection(self._entity_type, f'({self._expression}) AND NOT({self._get_relexpr(other)})')

    @property
    def as_tree(self):
        return query.parse_boolean_expression(self._expression)

    @property
    def entity_type(self) -> type[EntityT]:
        return self._entity_type

    @property
    def expression(self):
        return self._expression

    def _get_relexpr(self, other: str | DeferredEntityCollection):
        """Return expression string based on this object's entity_type"""
        if isinstance(other, DeferredEntityCollection):
            if other.entity_type == self.entity_type:
                expr = other.expression
            else:
                expr = f'[{other.entity_type.__name__}]{other.expression}'
            return expr

        return other

    def get_from(self, _entarchy: Entarchy) -> Collection[EntityT]:
        return self._entity_type.get_collection_type()(_entarchy, self._entity_type, _as_tree=self.as_tree)
