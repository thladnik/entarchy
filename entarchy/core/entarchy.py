from __future__ import annotations

import importlib
import os
import pathlib
import pprint
import sys
from typing import Any, Callable, Type, TYPE_CHECKING, Union

import alive_progress
import yaml

from . import query
from .entity import AnalysisEntity, Collection, EntarchyEntity, Entity, LinkEntity

if TYPE_CHECKING:
    from ..backend.backend import Backend
    from ..backend.mysql import MySQLBackend


class Entarchy(object):
    """Entarchy is a class that represents a system of hierarchically organized entities.
    """
    _base_version: str = '0.1'
    _base_compat_version_list: list[str] = ['0.1']
    _implementation_version: str
    _implementation_compat_version_list: list[str]
    _hierarchy_root_type: Type[Entity]
    max_blob_size: int = 10 * 1024 * 1024  # bytes (10MB)

    _backend: MySQLBackend = None  # Backend # TODO: Make this more generic, currently only MySQLBackend is supported
    _config: dict[str, Any] = None
    _is_in_context: bool = False
    _is_in_digest_mode: bool = False
    _current_analysis: Union[AnalysisEntity, None] = None
    _entarchy_entity: EntarchyEntity = None

    def __init__(self, path: str, debug: bool = False):
        self._path = pathlib.Path(path).absolute().as_posix()
        self._debug = debug

        # Resolve hierarchy
        self._hierarchy = {}
        self._entity_map = {}
        self._hierarchy, self._entity_map = self._resolve_hierarchy()

        # Load configuration from path
        self._config = yaml.safe_load(open(os.path.join(path, 'entarchy.yaml'), 'r'))

        if self._config['base_version'] not in self._base_compat_version_list:
            raise RuntimeError(f'Base version in configuration '
                               f'("{self._config["base_version"]}") '
                               f'is not compatible with current base version '
                               f'("{self._base_version}"). ')

        if self._config['implementation_version'] not in self._implementation_compat_version_list:
            raise RuntimeError(f'Implementation version in configuration '
                               f'("{self._config["implementation_version"]}") '
                               f'is not compatible with curreng implementation version '
                               f'("{self._implementation_version}"). ')

        if self._config['hierarchy'] != self._hierarchy:
            raise RuntimeError('Entity type hierarchy in configuration does not match the implementation. '
                               'This may be due to a corrupted configuration.')

        # Set up entities objects
        self._entities: dict[str, Entity] = {}
        self._entities_to_add: list[str] = []
        self._entities_to_update: list[str] = []
        self._links: dict[tuple[str, str], str] = {}

        self.roi_count = 0
        self.roi_attr_update_count = 0

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

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove backend from serialization, so it may stay open on original object
        #  This prevents broken file handles or database connections
        del state['_backend']
        return state

    def __hash__(self):
        return hash(self.path)

    def __repr__(self):
        return f'{self.__class__.__name__}(\'{self.path}\', backend={self.backend.__class__.__qualname__})'

    @property
    def backend(self) -> MySQLBackend:  # Backend:

        if not hasattr(self, '_backend') or self._backend is None:
            # Load backend
            _backend_path = self._config['backend']
            _backend_path_parts = _backend_path.split('.')
            _backend_cls = getattr(importlib.import_module('.'.join(_backend_path_parts[:-1])), _backend_path_parts[-1])
            self._backend = _backend_cls(**self._config['backend_config'], debug=self._debug)

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
    def hierarchy(self):
        return self._hierarchy.copy()

    @property
    def implementation_version(self):
        return self._implementation_version

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
                RuntimeError('No entarchy UUID found in configuration. Is the entarchy initialized correctly?')

            self._entarchy_entity = self.backend.get_entity_by_uuid(self, entarchy_uuid)

        return self._entarchy_entity

    @classmethod
    def _resolve_hierarchy(cls):

        def _resolve_hierarchy(entity_type, parent_dict, _entity_map):

            entity_name = entity_type.__name__

            if entity_name in _entity_map:
                raise ValueError(f'Entity type {entity_name} is already in the hierarchy. '
                                 f'Circular reference or duplicate name? Entity names must be unique.')

            _entity_map[entity_name] = entity_type

            # Go through children
            children = entity_type.get_child_entity_types()
            if entity_name not in parent_dict:
                parent_dict[entity_name] = {}
            if children is None:
                return
            for child_type in children:
                _resolve_hierarchy(child_type, parent_dict[entity_name], _entity_map)

        # Run and return result
        hierarchy = {'EntarchyEntity': {}, 'AnalysisEntity': {}, 'LinkEntity': {}}
        entity_map = {'EntarchyEntity': EntarchyEntity,'AnalysisEntity': AnalysisEntity, 'LinkEntity': LinkEntity}
        _resolve_hierarchy(cls._hierarchy_root_type, hierarchy, entity_map)

        return hierarchy, entity_map

    @classmethod
    def create(cls, path: str,
               _backend: Backend | str,
               _backend_config: dict[str, Any] = None,
               **kwargs) -> Entarchy:
        """Factory function to create an Entarchy instance from a given path.

        Args:
            path (str): The path to the entarchy configuration directory.
            _backend (Backend): an entarchy.backend.Backend object or its string representation
            _backend_config (dict): dictionary for initialization of backend if _backend is str or type
        Returns:
            Entarchy:
        """

        # Instantiate backend of necessary
        if isinstance(_backend, str):
            if '.' in _backend:
                _backend_path_parts = _backend.split('.')
                _backend_cls = getattr(importlib.import_module('.'.join(_backend_path_parts[:-1])), _backend_path_parts[-1])
            else:
                from .. import backend
                _backend_cls = getattr(backend, _backend)(**(_backend_config or {}), debug=False)

            _backend = _backend_cls(**(_backend_config or {}), debug=False)

        # Resolve hierarchy and add Analysis entity
        hierarchy, entity_map = cls._resolve_hierarchy()

        # Check if path exists, otherwise create
        if os.path.exists(path):
            raise FileExistsError(f'Path {path} already exists.')
        os.makedirs(path, exist_ok=False)

        print('---')
        print('Create entity type hierarchy:')
        pprint.pprint(hierarchy)
        print('---')

        # Save configuration so entarchy object can be created
        _config = {
            'base_version': cls._base_version,
            'implementation_version': cls._implementation_version,
            'backend': f'{_backend.__module__}.{_backend.__class__.__name__}',
            'backend_config': _backend.get_config(),
            'hierarchy': hierarchy
        }

        with open(os.path.join(path, 'entarchy.yaml'), 'w') as f:
            yaml.safe_dump(_config, f)

        # Create instance
        ent = cls(path, **kwargs)

        # Create backend
        res = ent.backend.create()
        if not res:
            raise RuntimeError('Failed to create backend.')

        # Create type hierarchy in backend
        res = ent.backend.create_type_hierarchy(hierarchy)
        if not res:
            raise RuntimeError('Failed to create entity type hierarchy in backend.')

        # Create entarchy entity
        with ent:
            ent_entity = EntarchyEntity(ent, _id='Entarchy', _parent=None)
            ent.add_new_entity(ent_entity)

        # Update config to include entarchy entity uuid
        with open(os.path.join(path, 'entarchy.yaml'), 'r+') as f:
            _config.update({'entarchy_uuid': ent_entity.uuid})
            yaml.safe_dump(_config, f)

        return ent

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

        # Add immutable id attributes
        entity['id'] = entity.id
        entity['uuid'] = entity.uuid
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

        # Add new entities
        if len(self._entities_to_add) > 0:

            res = self._backend.add_entities([self._entities[_uuid] for _uuid in self._entities_to_add])
            if not res:
                raise RuntimeError('Failed to add new entities to backend.')

            print(f'Added {len(self._entities_to_add)} entities')

            # Reset list
            self._entities_to_add = []

        # Commit updates for entities with attribute changes
        #  Note to future self: USE COPY, otherwise iterator is going to
        #  skip entries as the length of the list changes while updated elements are removed
        _entities_to_update = self._entities_to_update.copy()
        with alive_progress.alive_bar(monitor=f'| Update {len(_entities_to_update)} entities',
                                      monitor_end=f'Updated {len(_entities_to_update)} entities',
                                      bar=None, spinner='fish2', spinner_length=30, stats=False,
                                      force_tty=True) as bar:
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
            # return

        print(f'Deleting all data for analysis {path}')

        print('> Remove backend')
        self.backend.delete(True)

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

        # Add equality filters to filter expressions
        for k, v in _equalities.items():

            # Add quotes to string
            if isinstance(v, str):
                v = f'"{v}"'

            _string_expressions = _string_expressions + (f'{k} == {v}',)

        # Concat expression
        _expression = ' AND '.join(_string_expressions)

        # Parse to get dictionary representation
        as_tree = query.parse_boolean_expression(_expression)
        if as_tree is None:
            as_tree = {}

        # Return collection of custom type if available
        if entity_type.collection_type is not None:
            return entity_type.collection_type(self, entity_type, as_tree)

        # Fallback to generic collection
        return Collection(self, entity_type, as_tree)

    def get_config(self) -> dict[str, Any]:
        return self._config.copy()

    def get_entity(self, entity_type_name: str, _uuid: str, _id: str) -> Entity:

        return self.get_entity_type(entity_type_name)(self, _uuid=_uuid, _id=_id)

    def get_entity_by_uuid(self, _uuid: str) -> Entity:

        if _uuid in self._entities:
            return self._entities[_uuid]
        else:
            _entity = self.backend.get_entity_by_uuid(self, _uuid)

            return _entity

    def get_entity_type(self, entity_type_name: str) -> Type[Entity]:
        """Get the entity type class for a given entity type name.

        Args:
            entity_type_name (str): The name of the entity type.

        Returns:
            Type[Entity]: The entity type class.
        """

        # Get type from map
        entity_type = self._entity_map.get(entity_type_name, None)

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

    def remove_entity_from_update(self, entity: Entity) -> None:
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
                _analysis_entity = AnalysisEntity(self, _id=_analysis)
                self.add_new_entity(_analysis_entity)
                self.commit()
            else:
                _analysis_entity = AnalysisEntity(self, _id=_analysis)
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
