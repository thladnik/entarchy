from __future__ import annotations

import datetime
import uuid
from typing import Any, Iterable, Type, Union, TYPE_CHECKING

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
        self._entarchy.add_existing_entity(self)

        # Initialize cache if provided
        if _init_cache is not None:
            if not isinstance(_init_cache, dict):
                raise TypeError('_init_cache must be a dictionary of attribute names and values.')
            self._attribute_cache.update(_init_cache)

    def __new__(cls, _entarchy, _uuid=None, _id=None, _parent=None, _init_cache=None):

        if _uuid is not None and _uuid in _entarchy:
            obj = _entarchy.get_entity_by_uuid(_uuid)

            # Update cache if provided
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

        if not isinstance(item, (str, list, tuple)):
            raise TypeError('Item must be a string or list or tuple of strings.')

        if isinstance(item, (list, tuple)):
            if not all(isinstance(k, str) for k in item):
                raise TypeError('List or tuple of items must contain only strings.')

            return self._entarchy.backend.has_multiple_attributes(self._entarchy, self._uuid, item)

        return self._entarchy.backend.has_single_attribute(self._entarchy, self._uuid, item)

    def __enter__(self):
        # Set context flag
        self._is_in_context = True

        # Set current analysis if applicable
        if isinstance(self, Analysis):
            setattr(self, '__prev_analysis', self._entarchy.current_analysis)
            self._entarchy.set_current_analysis(self)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Commit any pending changes
        self.commit()

        # Reset context flag
        self._is_in_context = False

        # Reset current analysis if applicable
        if hasattr(self, '__prev_analysis'):
            self._entarchy.set_current_analysis(getattr(self, '__prev_analysis'))
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
            modified = self._entarchy.backend.get_entity_last_time_modified(self._entarchy, self.uuid)

            if modified > self._attribute_cache_start_time:
                self._attribute_cache_start_time = datetime.datetime.utcnow()
                self._attribute_cache = {}

        # Load missing attributes from backend
        keys_to_load = list(set(key) - set(self._attribute_cache.keys()))
        if len(keys_to_load) > 0:
            if len(keys_to_load) == 1:
                values = [
                    self._entarchy.backend.get_single_attribute_of_entity(self._entarchy, self.uuid, keys_to_load[0])]
            else:
                values = self._entarchy.backend.get_multiple_attributes_of_entity(self._entarchy, self.uuid, keys_to_load)

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

        self._entarchy.add_entity_for_update(self)

        # If not in context, update immediately
        if not self.is_in_context and not self._entarchy.is_in_context:
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
            str: The UUID string representaiton of the entity's PK
        """
        return str(self._uuid)

    @property
    def parent(self) -> Union[Entity, None]:
        """Get the parent entity of this entity.

        Returns:
            Entity: The parent entity.
        """
        return self._parent

    def commit(self):

        # Remove entity from entarchy update list
        self._entarchy.remove_entity_from_update(self)

        if len(self._attributes_to_update) > 0:

            names = self._attributes_to_update
            values = [self._attribute_cache[n] for n in names]
            if len(names) > 1:
                res = self._entarchy.backend.set_multiple_attributes_on_entity(self._entarchy,
                                                                               self.uuid,
                                                                               names,
                                                                               values)
            else:
                res = self._entarchy.backend.set_single_attribute_on_entity(self._entarchy,
                                                                            self.uuid,
                                                                            names[0],
                                                                            values[0])

            if not res:
                raise RuntimeError(f'Failed to update entity attributes {names} in backend.')

            # Reset list
            self._attributes_to_update = []


class Analysis(Entity):

    pass


class Collection(object):
    """Base class for collections of entities
    """

    _as_tree = None

    def __init__(self,
                 entity_type: Type[Entity],
                 _entarchy: Entarchy,
                 _query: Any = None):
        self._entity_type = entity_type
        self._entarchy = _entarchy
        self._as_tree = _query

        self._cache = pd.DataFrame()
        self._pending_changes: dict[str, list[int]] = {}

    def __len__(self):
        return self._entarchy.backend.get_entity_count_of_collection(self._entarchy,
                                                                     self.entity_type.__name__,
                                                                     self.as_tree)

    def __repr__(self):
        return f'Collection(entity_type=\'{self.entity_type.__name__}\', count={len(self)})'

    # Access methods

    def __getitem__(self, item):

        # Return single entity
        if isinstance(item, (int, np.integer)):
            if item < 0:
                item = len(self) + item

            _uuid, _id = self._entarchy.backend.get_entity_of_collection_by_index(self._entarchy,
                                                                                  self.entity_type.__name__,
                                                                                  self.as_tree, item)
            return self._get_entity(_uuid=_uuid, _id=_id)

        # Return slice
        elif isinstance(item, slice):

            # Get data
            res = self._entarchy.backend.get_entity_of_collection_by_slice(self._entarchy,
                                                                           self.entity_type.__name__,
                                                                           self.as_tree,
                                                                           item)

            result = [self._get_entity(_uuid=_uuid, _id=_id) for _uuid, _id in res]

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
        return Collection(self.entity_type, self._entarchy, new_tree)

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
        return Collection(self.entity_type, self._entarchy, new_tree)

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
        return Collection(self.entity_type, self._entarchy, new_tree)

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
        return Collection(self.entity_type, self._entarchy, new_tree)

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
        return Collection(self.entity_type, self._entarchy, new_tree)

    # Properties

    @property
    def as_tree(self) -> dict[str, ...]:
        """Return the abstract syntax tree dictionary
        """
        return self._as_tree.copy()

    @property
    def entity_type(self):
        return self._entity_type

    def _get_entity(self, _uuid: str, _id: str) -> Entity:

        _init_cache = None
        if _uuid in self._cache.index:
            _init_cache = self._cache.loc[_uuid].to_dict()

        return self.entity_type(_entarchy=self._entarchy, _uuid=_uuid, _id=_id, _init_cache=_init_cache)

    def _load_attributes(self, attribute_names: list[str]):

        # Load attributes from backend
        df = self._entarchy.backend.get_multiple_attributes_of_collection(self._entarchy,
                                                                          self.entity_type.__name__,
                                                                          self.as_tree,
                                                                          attribute_names)

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

    def update(self, df: pd.DataFrame):
        if not isinstance(df, pd.DataFrame):
            raise TypeError('df must be a pandas DataFrame.')

        # Update cache
        self._cache.update(df)

        # Send to backend
        self._entarchy.backend.set_multiple_attributes_on_collection(self._entarchy,
                                                                     self.entity_type.__name__,
                                                                     self.as_tree,
                                                                     df)
