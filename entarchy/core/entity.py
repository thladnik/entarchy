from __future__ import annotations
import uuid
from typing import Any, Type, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .analysis import Analysis
    from .entarchy import Entarchy


class Entity(object):
    """Base class for all entities in the Entarchy system.
    This class can be extended to create specific types of entities.
    """

    _child_entity_types: list[Type[Entity]] = None
    _is_in_context: bool = False

    def __init__(self, _entarchy: Entarchy,
                 _analysis: Analysis = None, _uuid: str = None, _id: str = None, _parent: Entity = None):
        self._entarchy = _entarchy
        self._analysis = _analysis
        self._uuid = _uuid
        self._id = _id
        self._parent = _parent

        if self._uuid is None and self._id is None:
            raise ValueError("Need to provide either _pk or _id")

        # Create PK if not provided
        if self._uuid is None:
            self._uuid = uuid.uuid4()

        # Set up attribute cachse
        self._attribute_cache: dict[str, Any] = {}
        self._attributes_to_update: list[str] = []

        # Add entity to entarchy object
        self._entarchy.add_existing_entity(self)

    def __enter__(self):
        # Set context flag
        self._is_in_context = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Commit any pending changes
        self.commit()
        # Reset context flag
        self._is_in_context = False

    def __repr__(self):
        return f'{self.__class__.__name__}(id="{self.id}" parent="{self.parent}")'

    def __setitem__(self, key: Union[str, list[str], tuple[str, ...]] , value: Any):
        """Set a dynamic attribute on the entity.

        Dynamic attribute keys must always be strings.
        Using a list or tuple of strings will expect the values to be a list or tuple of the same length.

        Args:
            key (str or list/tuple of str): The key(s) for the attribute(s) to set.
            value (Any or list/tuple of Any): The value(s) for the attribute(s) to set.

        Raises:
            TypeError: If key is not a string or list/tuple of strings.
            TypeError: If value is not a list/tuple when key is a list/tuple.
            ValueError: If lengths of key and value lists/tuples do not match.

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
        if not self._is_in_context:
            self.commit()

    def __getitem__(self, key: Union[str, list[str]]) -> Union[Any, tuple[Any, ...]]:
        """Get a dynamic attribute from the entity.

        Dynamic attribute keys must always be strings.
        Using a list or tuple of strings will return a list or tuple of the same length.

        Args:
            key (str or list/tuple of str): The key(s) for the attribute(s) to get.

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

        # Reset attribute cache if entity is dirty
        if self._entarchy.entity_is_dirty(self):
            self._attribute_cache = {}

        # Load missing attributes from backend
        keys_to_load = [k for k in key if k not in self._attribute_cache]
        if len(keys_to_load) > 1:
            values = self._entarchy.backend.get_multiple_attributes_of_entity(self._entarchy, self._analysis, self._uuid, keys_to_load)
        else:
            values = [self._entarchy.backend.get_single_attribute_of_entity(self._entarchy, self._analysis, self._uuid, keys_to_load[0])]

        # Update cache
        for k, v in zip(keys_to_load, values):
            self._attribute_cache[k] = v

        return (self._attribute_cache[k] for k in key)

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

            return self._entarchy.backend.has_multiple_attributes(self._entarchy, self._analysis, self._uuid, item)

        return self._entarchy.backend.has_single_attribute(self._entarchy, self._analysis, self._uuid, item)

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

        if len(self._attributes_to_update) == 0:
            return

        names = self._attributes_to_update
        values = [self._attribute_cache[n] for n in names]
        if len(names) > 1:
            res = self._entarchy.backend.set_multiple_attributes_on_entity(self._entarchy, self._analysis, self._uuid, names, values)
        else:
            res = self._entarchy.backend.set_single_attribute_on_entity(self._entarchy, self._analysis, self._uuid, names[0], values[0])

        if not res:
            raise RuntimeError(f'Failed to update entity attributes {names} in backend.')

        # Reset list
        self._attributes_to_update = []
        # Remove entity from entarchy update list
        self._entarchy.remove_entity_from_update(self)


class Collection(object):
    """Base class for collections of entities
    """
    pass
