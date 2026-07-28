from __future__ import annotations

import atexit
import datetime
import time
import traceback
import uuid
from typing import Any, Generator, Type, Union, TYPE_CHECKING, Callable

import alive_progress
import numpy as np
import pandas as pd

from . import console
from . import query

if TYPE_CHECKING:
    from .entarchy import Entarchy


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
                 _uuid: str = None,
                 _id: str = None,
                 _parent: Entity = None,
                 _init_cache: dict[str, Any] = None):

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

    def __new__(cls, *args, **kwargs):

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

    def __matmul__(self, other) -> LinkEntity:
        """Create a link entity between this entity and another entity.

        Args:
            other (Entity): The target entity to link to.
        """

        raise NotImplementedError('Links are not implemented yet.')

        if not isinstance(other, Entity):
            raise TypeError('Can only create link to another Entity instance.')

        link_entity = LinkEntity(_entarchy=self.entarchy)

        self.entarchy.add_new_entity(link_entity)

        return link_entity

    def __repr__(self):
        return f'{self.__class__.__name__}(id=\'{self.id}\' uuid=\'{self.uuid}\')'

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

            # Reset list
            self._attributes_to_update = []

        # In digest mode, purge cache from memory after commit
        if self.entarchy.is_in_digest_mode:
            self._attribute_cache = {}

    def keys(self):
        """Return a list of all dynamic attribute keys for this entity.
        """
        return self.entarchy.backend.get_entity_attribute_names(self)

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
    """Analysis entity for data segregation
    """


class LinkEntity(Entity):
    """Link entity between two entities
    """


class Collection(object):
    """Base class for collection of entities
    """

    # TODO: implement .apply method along column axis similar to pandas DataFrame.apply() ?

    _as_tree: dict[str, ...]
    _length: int = None
    _name = None

    def __init__(self,
                 _entarchy: Entarchy,
                 _entity_type: Type[Entity],
                 _as_tree: dict[str, ...]):
        self._entity_type = _entity_type
        self._entarchy = _entarchy
        self._as_tree = _as_tree

        self._cache = pd.DataFrame()
        self._pending_changes: dict[str, list[int]] = {}
        self._init_time = datetime.datetime.now()

    def __len__(self):
        if self._length is None:
            self._length = self.entarchy.backend.get_collection_count(self)
        return self._length

    def __repr__(self):
        if self.name is not None:
            return self.name
        if self.__class__.__name__ == 'Collection':
            return f'{self.__class__.__name__}(entity_type=\'{self.entity_type.__name__}\', count={len(self)})'
        return f'{self.__class__.__name__}(count={len(self)})'

    # Access methods

    def __getitem__(self, item):

        # Return single entity
        if isinstance(item, (int, np.integer)):
            if item < 0:
                item = len(self) + item

            _uuid, _id = self.entarchy.backend.get_collection_entity_by_index(self, item)
            return self.get_entity(_uuid=_uuid, _id=_id)

        # Return slice
        elif isinstance(item, slice):

            # Get data
            res = self.entarchy.backend.get_collection_entities_by_slice(self, item)

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

    def __iter__(self):
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

    def __or__(self, other: Collection | str):
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
        return Collection(self.entarchy, self.entity_type, new_tree)

    def __and__(self, other: Collection | str):
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
        return Collection(self.entarchy, self.entity_type, new_tree)

    def __invert__(self):
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
        return Collection(self.entarchy, self.entity_type, new_tree)

    def __sub__(self, other: Collection | str):
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
        return Collection(self.entarchy, self.entity_type, new_tree)

    def __xor__(self, other: Collection | str):
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
        return Collection(self.entarchy, self.entity_type, new_tree)

    # Properties

    @property
    def as_tree(self) -> dict[str, ...]:
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
    def entity_type(self):
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
            uuids, parent_uuids = list(zip(*self.entarchy.backend.get_collection_parent_uuids(self)))
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

                # Create a list of parent values
                parent_values = []
                for parent_uuid in parent_uuids:

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

            return df[original_attribute_order]

        df = self._cache[attribute_names].copy()

        if uuid_requested:
            df['uuid'] = df.index

        return df[original_attribute_order]

        # TODO: return final DataFrame in custom order
        # if self._query_custom_orderby:
        #     return self._cache.loc[self._pk_order, attribute_names]

    def get_entity(self, _uuid: str, _id: str) -> Entity:

        _init_cache = None
        if _uuid in self._cache.index:
            _init_cache = self._cache.loc[_uuid].to_dict()

        return self.entity_type(_entarchy=self.entarchy, _uuid=_uuid, _id=_id, _init_cache=_init_cache)

    def keys(self) -> list[str]:
        return self.columns

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
                  _locality: bool = True,
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
            **kwargs: Passed on to the function.

        Returns:
            None
        """

        import multiprocessing as mp

        # Address entities by UUID rather than shipping pickled Entity objects
        entity_rows = self.entarchy.backend.get_collection_entities_by_slice(self, slice(None, None, None))
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

        # Group entities by parent, so a worker sees runs of entities that share
        #  the (often large) attributes of their parent instead of jumping between
        #  parents in UUID order. Entities are stored with random UUID4 keys, so
        #  the unsorted order has no locality at all.
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
                pool, was_running = _get_worker_pool(self.entarchy, _worker_num)
                print(f'{"Reuse" if was_running else "Start"} worker pool '
                      f'({_worker_num} workers, spawn, chunk size {_chunk_size})')

                kwargs_items = tuple(kwargs.items())
                tasks = [(_uuid, _id, parent_uuids.get(_uuid), self.entity_type,
                          _fun, kwargs_items, gpu_device_count)
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

    def to_dict(self) -> Generator[dict[str, Any]]:
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

    def where(self, *_string_expressions: str, **_equalities):
        _collection = self.entarchy.get(self.entity_type, *_string_expressions, **_equalities)
        if self.as_tree == {}:
            new_tree = _collection.as_tree
        else:
            new_tree = query.combine_trees('INTERSECTION', self.as_tree, _collection.as_tree)

        if self.entity_type.collection_type is None:
            collection_type = Collection
        else:
            collection_type = self.entity_type.collection_type

        return collection_type(self.entarchy, self.entity_type, new_tree)


# Per-worker state for Collection.map_async.
#  Built once per worker process by _init_map_worker, so that the Entarchy object
#  and its database connection are reused across all tasks handled by that worker.
_WORKER_CONTEXT: dict[str, Any] = {}

# At most one reusable worker pool is kept alive, so repeated map_async calls do
#  not each pay the pool startup cost.
_POOL_CACHE: dict[str, Any] = {}

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
    _uuid, _id, parent_uuid, entity_type, fun, kwargs_items, gpu_device_count = task

    _entarchy = _WORKER_CONTEXT['entarchy']
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


def _get_worker_pool(_entarchy: Entarchy, worker_num: int) -> tuple:
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
                    initargs=(_entarchy, counter))

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


class CollectionIterator(object):

    def __init__(self, _collection: Collection):
        self._collection = _collection
        self._current_index = 0
        self._results = self._collection.entarchy.backend.get_collection_entities_by_slice(self._collection,
                                                                                           slice(None, None, None))

    def __next__(self):
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


class DeferredEntityCollection(object):
    _expression: str

    def __init__(self, _entity_type: type[Entity], _expression: str = ''):
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
    def entity_type(self):
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

    def get_from(self, _entarchy: Entarchy) -> Collection:
        return self._entity_type.get_collection_type()(_entarchy, self._entity_type, _as_tree=self.as_tree)
