from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Generator, Type, Union, TYPE_CHECKING, Callable

import alive_progress
import numpy as np
import pandas as pd

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

    def __new__(cls, *args, **kwargs):

        _entarchy = args[0] if len(args) > 0 else kwargs.get('_entarchy', None)
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
        # Commit any pending changes
        self.commit()

        # Reset context flag
        self._is_in_context = False

        # Reset current analysis if applicable
        if hasattr(self, '__prev_analysis'):
            self.entarchy.set_current_analysis(getattr(self, '__prev_analysis'))
            delattr(self, '__prev_analysis')

    def __hash__(self):
        return self.uuid

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

        # Update attribute(s) in cache and mark for update
        for k, v in zip(key, value):

            self._attribute_cache[k] = v

            if k not in self._attributes_to_update:
                self._attributes_to_update.append(k)

        self.entarchy.add_entity_for_update(self)

        # If not in context, update immediately
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
    pass


class AnalysisEntity(Entity):
    pass


class LinkEntity(Entity):
    pass


class Collection(object):
    """Base class for collections of entities
    """

    # TODO: implement .apply method along column axis similar to pandas DataFrame.apply() ?

    _as_tree: dict[str, ...]
    _length: int = None

    def __init__(self,
                 _entarchy: Entarchy,
                 entity_type: Type[Entity],
                 _as_tree: dict[str, ...]):
        self._entity_type = entity_type
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
        return f'{self.__class__.__name__}(entity_type=\'{self.entity_type.__name__}\', count={len(self)})'

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

    def _load_attributes(self, attribute_names: list[str]):

        # Load attributes from backend
        df = self.entarchy.backend.get_collection_attributes(self, attribute_names)

        # Update cache
        self._cache[df.columns] = df

    def dataframe_of(self, attribute_names: list[str] = None, reload_cached: bool = False) -> pd.DataFrame:

        original_attribute_order = attribute_names.copy()

        # Check for parent attributes in attribute_names and separate them from regular attributes,
        #  because they need to be fetched separately
        parent_attribute_names = [n for n in attribute_names if n.startswith('../') or n.startswith('[')]
        attribute_names = list(set(attribute_names) - set(parent_attribute_names))

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

            return df[original_attribute_order]

        return self._cache[original_attribute_order].copy()

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

    def map(self, fun: Callable, **kwargs) -> Any:
        """Sequentially apply a function to each Entity of the collection (kwargs are passed onto the function)
        """

        entity_count = len(self)

        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {entity_count} entities')

        with alive_progress.alive_bar(entity_count, spinner='fish2', force_tty=True) as bar:
            for entity in self:
                fun(entity, **kwargs)
                bar()

    def map_async(self,
                  _fun: Callable,
                  _worker_num: int = None,
                  _chunk_size: int = None,
                  _use_gpu: bool = False,
                  _gpu_max_device_num: int = None,
                  **kwargs) -> None:
        """
        Concurrently apply a function to each Entity of the collection (kwargs are passed onto the function)

        worker_num: int number of subprocess workers to spawn for parallel execution
        chunk_size: int (optional) size of chunks for batched execution of function to decrease overhead
            (note that for batch execution the first argument
            of fun is going to be of type List[Entity] instead of Entity)
        """

        entity_count = len(self)

        print(f'Run function {_fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {entity_count} {self.entity_type.__name__} entities')

        if entity_count == 0:
            print('No entities to operate on in collection')
            return

        # Prepare pool and entities
        import multiprocessing as mp
        if _worker_num is None:
            _worker_num = mp.cpu_count() - 2
            if entity_count < _worker_num:
                _worker_num = entity_count

        # Set chunk size (crucial for performance of large collections)
        if _chunk_size is None:
            _chunk_size = int(entity_count / 1_000)
            if _chunk_size == 0:
                _chunk_size = 1

        # Set _gpu_max_device_num if necessary
        if _use_gpu and _gpu_max_device_num is None:
            import torch
            kwargs['_use_gpu'] = _use_gpu
            kwargs['_gpu_max_device_num'] = torch.cuda.device_count()

        ctx_type = 'spawn'
        ctx = mp.get_context(ctx_type)
        # mp.set_start_method(ctx_type, force=True)
        print(f'Start pool ({ctx_type}) with {_worker_num} workers')
        with ctx.Pool(processes=_worker_num) as pool:

            # Map entities to process pool
            start_time = time.time()
            print(f'Start processing at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')

            t = time.perf_counter()
            # iterator = pool.imap_unordered(self.worker_wrapper, self._async_iterator(_fun, **kwargs), chunksize=_chunk_size)
            iterator = pool.imap(self.worker_wrapper, self._async_iterator(_fun, **kwargs), chunksize=_chunk_size)

            with alive_progress.alive_bar(entity_count,
                                          spinner='fish2', spinner_length=20,
                                          length=20, force_tty=True) as progress_bar:

                for iter_num in range(entity_count):

                    _ = next(iterator)

                    # print('Next')

                    progress_bar()

            pool.close()
            pool.join()

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(f'\nFinish processing at {formatted_time}')

    def _async_iterator(self, fun: Callable, **kwargs) -> Generator[Any, None, None]:
        """Generator that yields results of applying a function to each entity in the collection concurrently.
        """
        kwargs = tuple([(k, v) for k, v in kwargs.items()])
        for i, entity in enumerate(self):
            yield fun, i, entity, kwargs

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

        # Update cache
        self._cache.update(df)

        # Send to backend
        self.entarchy.backend.set_collection_attributes(self, df)

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

    @staticmethod
    def worker_wrapper(args):
        """Subprocess wrapper function for concurrent execution, which handles the MySQL session
        and provides feedback on execution time to parent process
        """

        start_time = time.perf_counter()

        # Unpack args
        fun: Callable = args[0]
        i: int = args[1]
        entity: Entity = args[2]
        kwargs = {k: v for k, v in args[3]}

        # Set GPU device if applicable
        _use_gpu = kwargs.pop('_use_gpu', False)
        if _use_gpu:
            _gpu_max_device_num = kwargs.pop('_gpu_max_device_num')
            if _gpu_max_device_num > 0:
                _use_gpu_device = f'cuda:{i % _gpu_max_device_num}'
            else:
                _use_gpu_device = 'cpu'

            kwargs['_use_gpu_device'] = _use_gpu_device

        # Run
        with entity:
            fun(entity, **kwargs)

        # Close backend to avoid broken file handles and dangling open database connections in subprocesses after fork
        #  (especially important for MySQL backend, which does not allow sharing connections between processes)
        entity.entarchy.backend.close()

        return time.perf_counter() - start_time


class CollectionIterator(object):

    def __init__(self, _collection: Collection):
        self._collection = _collection
        self._current_index = 0
        self._results = self._collection.entarchy.backend.get_collection_entities_by_slice(self._collection, slice(None, None, None))

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