from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Type, Union, TYPE_CHECKING, Callable

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
        self._attribute_cache_start_time = datetime.datetime.utcnow()
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
            _init_cache = kwargs.get('_init_cache', None)
            if _init_cache is not None:
                obj.update_cache(_init_cache)

            return obj

        return super().__new__(cls)

    def __contains__(self, item: Union[str, list[str]]) -> bool:
        """Check if the entity has a dynamic attribute.

        Dynamic attribute keys must always be strings.
        Using a list or tuple of strings will check if all attributes exist.

        Args:
            item (str or list/tuple of str): The key(s) for the attribute(s) to check.

        Raises:
            TypeError: If item is not a string or list/tuple of strings.

        Returns:
            bool: True if the attribute(s) exist, False otherwise.
        """

        # TODO: also refer to cache
        raise NotImplementedError('')

        if not isinstance(item, (str, list, tuple)):
            raise TypeError('Item must be a string or list or tuple of strings.')

        if isinstance(item, (list, tuple)):
            if not all(isinstance(k, str) for k in item):
                raise TypeError('List or tuple of items must contain only strings.')

            return self.entarchy.backend.has_multiple_attributes(self, item)

        return self.entarchy.backend.has_single_attribute(self, item)

    def __enter__(self):
        # Set context flag
        self._is_in_context = True

        # Set current analysis if applicable
        if isinstance(self, Analysis):
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

        # Reset attribute cache if any key is in cache and entity is dirty
        if any([k in self._attribute_cache for k in key]):
            modified = self.entarchy.backend.get_entity_modified_time(self)

            if modified > self._attribute_cache_start_time:
                self._attribute_cache_start_time = datetime.datetime.utcnow()
                self._attribute_cache = {}

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

    def commit(self):

        # Remove entity from entarchy update list
        self.entarchy.remove_entity_from_update(self)

        if len(self._attributes_to_update) > 0:

            names = self._attributes_to_update
            values = [self._attribute_cache[n] for n in names]
            if len(names) > 1:
                res = self.entarchy.backend.set_entity_attributes(self, names, values)
            else:
                res = self.entarchy.backend.set_entity_attribute(self, names[0], values[0])

            if not res:
                raise RuntimeError(f'Failed to update entity attributes {names} in backend.')

            # Reset list
            self._attributes_to_update = []


class Analysis(Entity):

    pass


class Collection(object):
    """Base class for collections of entities
    """

    _as_tree: dict[str, ...]

    def __init__(self,
                 _entarchy: Entarchy,
                 entity_type: Type[Entity],
                 _as_tree: dict[str, ...]):
        self._entity_type = entity_type
        self._entarchy = _entarchy
        self._as_tree = _as_tree

        self._cache = pd.DataFrame()
        self._pending_changes: dict[str, list[int]] = {}
        self._init_time = datetime.datetime.utcnow()

    def __len__(self):
        return self.entarchy.backend.get_collection_count(self)

    def __repr__(self):
        return f'Collection(entity_type=\'{self.entity_type.__name__}\', count={len(self)})'

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
        return CollectionBatchIterator(self)

    def __setitem__(self, key, value):

        if key != slice(None, None, None):
            raise KeyError(f'Invalid key {key}')
        if not isinstance(value, pd.DataFrame):
            if not isinstance(value, pd.Series):
                raise ValueError(f'Invalid value of type {type(value)}. Needs to be pandas.Series or pandas.DataFrame')
            value = pd.DataFrame(value)

        self.update(value)

    # Set operations

    def __add__(self, other):
        """Create a new collection that is the union between this collection and another.

        Args:
            other (Collection): Another collection to add.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a + collection_b
        """

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

    def __and__(self, other):
        """Create a new collection that is the intersection between this collection and another.

        Args:
            other (Collection): Another collection to intersect.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a & collection_b
        """

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

    def __sub__(self, other):
        """Create a new collection that is the difference between this collection and another.

        Args:
            other (Collection): Another collection to subtract.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a - collection_b
        """

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

    def __xor__(self, other):
        """Create a new collection that is the symmetric difference between this collection and another.

        Args:
            other (Collection): Another collection to xor.

        Example:
            collection_a = Collection(EntityTypeA, entarchy, query_a)
            collection_b = Collection(EntityTypeA, entarchy, query_b)
            collection_c = collection_a ^ collection_b
        """

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
    def entarchy(self) -> Entarchy:
        return self._entarchy

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def init_time(self) -> datetime.datetime:
        return self._init_time

    def get_entity(self, _uuid: str, _id: str) -> Entity:

        _init_cache = None
        if _uuid in self._cache.index:
            _init_cache = self._cache.loc[_uuid].to_dict()

        return self.entity_type(_entarchy=self.entarchy, _uuid=_uuid, _id=_id, _init_cache=_init_cache)

    def _load_attributes(self, attribute_names: list[str]):

        # Load attributes from backend
        df = self.entarchy.backend.get_collection_attributes(self, attribute_names)

        # Update cache
        self._cache[df.columns] = df

    def dataframe_of(self, attribute_names: list[str] = None, reload_cached: bool = False) -> pd.DataFrame:

        # If all attributes are in cache, return cached result
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

            # Load attributes from database
            self._load_attributes(_attributes_to_fetch)

        # Return final DataFrame
        # if self._query_custom_orderby:
        #     return self._cache.loc[self._pk_order, attribute_names]

        return self._cache[attribute_names].copy()

    def map(self, fun: Callable, **kwargs) -> Any:
        """Sequentially apply a function to each Entity of the collection (kwargs are passed onto the function)
        """

        entity_count = len(self)

        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {entity_count} entities')

        with alive_progress.alive_bar(entity_count, spinner='fishes') as bar:
            for entity in self:
                fun(entity, **kwargs)
                bar()

    def map_async(self, fun: Callable, worker_num: int = None, **kwargs) -> Any:
        """Concurrently apply a function to each Entity of the collection (kwargs are passed onto the function)

        worker_num: int number of subprocess workers to spawn for parallel execution
        chunk_size: int (optional) size of chunks for batched execution of function to decrease overhead
            (note that for batch execution the first argument
            of fun is going to be of type List[Entity] instead of Entity)
        """

        entity_count = len(self)

        print(f'Run function {fun.__name__} on {self} with args '
              f'{[f"{k}:{v}" for k, v in kwargs.items()]} on {entity_count} {self.entity_type.__name__} entities')

        if len(self) == 0:
            print('No entities to operate on in collection')
            return

        # Prepare pool and entities
        import multiprocessing as mp
        if worker_num is None:
            worker_num = mp.cpu_count() - 2
            if entity_count < worker_num:
                worker_num = entity_count
        print(f'Start pool with {worker_num} workers')

        # Package entities together with their mapped function and arguments
        #  and make the entity table instances transient
        print(f'Prepare entities')
        t = time.perf_counter()
        kwargs = tuple([(k, v) for k, v in kwargs.items()])
        worker_args = []
        for entity in self:
            worker_args.append((fun, entity, kwargs))

        print(f'> Preparation finished in {time.perf_counter() - t:.2f}s')

        # Map entities to process pool
        start_time = time.time()
        print(f'Start processing at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))}')
        self.entarchy.backend.close()
        with (mp.Pool(processes=worker_num, initializer=self.worker_init, initargs=(self.entarchy,)) as pool,
              alive_progress.alive_bar(entity_count, spinner='fishes') as bar):

            iterator = pool.imap_unordered(self.worker_wrapper, worker_args)
            for iter_num in range(entity_count):

                # Next iteration
                try:
                    exec_time = next(iterator)

                # Catch
                except StopIteration:
                    pass

                # Re-raise any exception raised by worker wrapper
                except Exception as _exc:
                    raise _exc

                # Calcualate timing info
                # execution_times.append(exec_time)
                # mean_exec_time = np.mean(execution_times) if len(execution_times) > 0 else 0
                # time_per_entity = mean_exec_time / worker_num
                # time_elapsed = time.time() - start_time
                # time_rest = time_per_entity * (entity_count - iter_num)

                bar()

                # pbar.update(1)
                # pbar.set_postfix({
                #     'time_per_iter': f'{time_per_entity:.2f}s',
                #     'elapsed': str(datetime.timedelta(seconds=int(time_elapsed))),
                #     'eta': str(datetime.timedelta(seconds=int(time_rest))),
                # })

        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
        print(f'\nFinish processing at {formatted_time}')

    def update(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas DataFrame.')

        # Update cache
        self._cache.update(df)

        # Send to backend
        self.entarchy.backend.set_collection_attributes(self, df)

    def where(self, *_string_expressions: str, **_equalities):
        _collection = self.entarchy.get(self.entity_type, *_string_expressions, **_equalities)
        new_tree = query.combine_trees('INTERSECTION', self.as_tree, _collection.as_tree)

        return Collection(self.entarchy, self.entity_type, new_tree)

    @staticmethod
    def worker_init(_entarchy: Entarchy):
        """Subprocess initializer function for concurrent execution
        """

        _entarchy.backend.open()

    @staticmethod
    def worker_wrapper(args):
        """Subprocess wrapper function for concurrent execution, which handles the MySQL session
        and provides feedback on execution time to parent process
        """

        start_time = time.perf_counter()

        # Unpack args
        fun: Callable = args[0]
        entity: Entity = args[1]
        kwargs = {k: v for k, v in args[2]}

        # Run
        fun(entity, **kwargs)

        return time.perf_counter() - start_time


class CollectionBatchIterator(object):

    def __init__(self, _collection: Collection):
        self._collection = _collection
        self._batch_size = 100
        self._current_index = 0
        self._total_length = len(_collection)
        self._batch_offset = 0
        self._batch_results = []

    def __next__(self):
        # Fetch the next batch of results
        if self._current_index == 0 or (self._current_index == self._batch_offset + self._batch_size):
            self._batch_offset = self._current_index
            _slice = slice(self._batch_offset, self._batch_offset + self._batch_size)
            res = self._collection.entarchy.backend.get_collection_entities_by_slice(self._collection, _slice)

            self._batch_results = res

        # No more results: reset iteration counter and offset and stop iteration
        if len(self._batch_results) == 0 or self._current_index >= self._total_length:
            raise StopIteration

        # Return single result
        current_row = self._batch_results[self._current_index - self._batch_offset]

        # Increment count
        self._current_index += 1

        return self._collection.get_entity(_uuid=current_row[0], _id=current_row[1])
