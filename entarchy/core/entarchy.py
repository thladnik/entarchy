from __future__ import annotations

import abc
import importlib
import os
import pathlib
import pprint
import sys
from typing import Any, Type, TYPE_CHECKING

import yaml

from . import query
from .entity import Collection, Entity

if TYPE_CHECKING:
    from ..backend.backend import Backend


class Entarchy:
    """Entarchy is a class that represents a system of hierarchically organized entities.
    """
    _base_version: str = '0.1'
    _implementation_version: str

    _hierarchy_root: Type[Entity]

    _backend: Backend = None
    _config: dict[str, Any] = None
    _is_in_context: bool = False

    def __init__(self, path: str, debug: bool = False):
        self._path = pathlib.Path(path).absolute().as_posix()
        self._debug = debug

        # Load configuration from path
        self._config = yaml.safe_load(open(os.path.join(path, 'entarchy.yaml'), 'r'))

        # Load backend
        _backend_path = self._config['backend']
        _backend_path_parts = _backend_path.split('.')
        _backend_cls = getattr(importlib.import_module('.'.join(_backend_path_parts[:-1])), _backend_path_parts[-1])
        self._backend = _backend_cls(**self._config['backend_config'], debug=self._debug)

        # Set up entities objects
        self._entities: dict[str, Entity] = {}
        self._entities_to_add: list[str] = []
        self._entities_to_update: list[str] = []

        print(f'{self} opened with backend {_backend_path}')

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
        self.commit()
        self._is_in_context = False

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return f'Entarchy(\'{self.path}\')'

    @property
    def backend(self) -> Backend:
        return self._backend

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
    def is_in_context(self) -> bool:
        """Check if the entarchy is in a context manager.

        Returns:
            bool: True if the entarchy is in a context manager, False otherwise.
        """
        return self._is_in_context

    @property
    def path(self) -> str:
        return self._path

    def get_config(self) -> dict[str, Any]:
        return self._config.copy()

    @classmethod
    def create(cls, path: str, _backend: Backend, *args, **kwargs) -> Entarchy:
        """Factory function to create an Entarchy instance from a given path.

        Args:
            path (str): The path to the entarchy configuration directory.
            _backend (Backend): The backend instance to use for this entarchy.

        Returns:
            Entarchy: An instance of cls.
        """

        # Parse hierarchy
        _hierarchy = {}

        def _add_to_hierarchy(entity_type, parent_dict):
            children = entity_type.get_child_entity_types()
            if entity_type.__name__ not in parent_dict:
                parent_dict[entity_type.__name__] = {}
            if children is None:
                return
            for child_type in children:
                _add_to_hierarchy(child_type, parent_dict[entity_type.__name__])

        _add_to_hierarchy(cls._hierarchy_root, _hierarchy)

        # Check if path exists, otherwise create
        if os.path.exists(path):
            raise FileExistsError(f'Path {path} already exists.')
        os.makedirs(path, exist_ok=False)

        print('---')
        print('Create entity type hierarchy:')
        pprint.pprint(_hierarchy)
        print('---')

        _config = {
            'base_version': cls._base_version,
            'implementation_version': cls._implementation_version,
            'backend': f'{_backend.__module__}.{_backend.__class__.__name__}',
            'backend_config': _backend.get_config(),
            'hierarchy': _hierarchy
        }

        with open(os.path.join(path, 'entarchy.yaml'), 'w') as f:
            yaml.safe_dump(_config, f)

        # Create instance
        entarchy = cls(path, *args, **kwargs)

        # Create backend
        res = entarchy.backend.create()
        if not res:
            raise RuntimeError('Failed to create backend.')

        # Create type hierarchy in backend
        res = entarchy.backend.create_type_hierarchy(_hierarchy)
        if not res:
            raise RuntimeError('Failed to create entity type hierarchy in backend.')

        return entarchy

    def add_entity_for_update(self, entity: Entity) -> None:
        """Mark an entity for update in the backend.

        Args:
            entity (Entity): The entity to mark for update.

        Returns:
            None
        """

        if entity.uuid not in self._entities_to_update:
            self._entities_to_update.append(entity.uuid)

    def add_new_entity(self, entity: Entity) -> None:
        """Add an entity to the entarchy system.

        Args:
            entity (Entity): The entity to add.

        Returns:
            None
        """
        self.add_existing_entity(entity)
        self._entities_to_add.append(entity.uuid)

    def add_existing_entity(self, entity: Entity) -> None:
        """Add an entity to the entarchy system.

        Args:
            entity (Entity): The entity to add.

        Returns:
            None
        """
        self._entities[entity.uuid] = entity

    def commit(self) -> None:
        """Commit new entities to the backend.

        Returns:
            None
        """
        if len(self._entities_to_add) == 0:
            return

        # Add new entities
        res = self._backend.add_entities([self._entities[_uuid] for _uuid in self._entities_to_add])
        if not res:
            raise RuntimeError('Failed to add new entities to backend.')

        self._entities_to_add = []

        for _uuid in self._entities_to_update:
            self._entities[_uuid].commit()

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
            # return

        print(f'Deleting all data for analysis {path}')

        print('> Remove backend')
        self.backend.delete(True)

        print('> Remove directories and files')
        # Use pathlib recursive unlinker by mitch from https://stackoverflow.com/a/49782093
        def rmdir(directory, counter):
            directory = pathlib.Path(directory)
            for item in directory.iterdir():
                if item.is_dir():
                    counter = rmdir(item, counter)
                else:
                    item.unlink()

            if counter % 10 == 0:
                # print(' ' * 500, end='\n')
                sys.stdout.write(f'\rRemove {directory.as_posix()}')
            counter += 1
            directory.rmdir()

            return counter

        # Delete tree
        rmdir(path, 0)

        print(f'\nSuccessfully deleted analysis {path}')

    @abc.abstractmethod
    def digest(self, raw_data_path: str) -> None:
        """Digest raw data from a given path into the entarchy system.

        Args:
            raw_data_path (str): The path to the raw data to digest.

        Returns:
            None
        """
        pass

    def get(self, entity_type: Type[Entity], *_string_expressions: str, **_equalities) -> Collection:
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

        # Add equality filters to filter expressions
        for k, v in _equalities.items():

            # Add quotes to string
            if isinstance(v, str):
                v = f'"{v}"'

            _string_expressions = _string_expressions + (f'{k} == {v}',)

        # Concat expression
        _expression = ' AND '.join(_string_expressions)

        # Parse to get dictionary representation
        _parsed_query = query.parse_boolean_expression(_expression)
        if _parsed_query is None:
            _parsed_query = {}

        return Collection(entity_type=entity_type, _entarchy=self, _query=_parsed_query)

    def get_entity_by_uuid(self, _uuid: str) -> Entity:

        if _uuid in self._entities:
            return self._entities[_uuid]
        else:
            entity_data = self.backend.get_entity_by_uuid(self, _uuid)
            return Entity(_entarchy=self, _uuid=entity_data[0], _id=entity_data[1])

    def remove_entity_from_update(self, entity: Entity) -> None:
        """Unmark an entity for update in the backend.

        Args:
            entity (Entity): The entity to unmark for update.

        Returns:
            None
        """

        if entity.uuid in self._entities_to_update:
            self._entities_to_update.remove(entity.uuid)
